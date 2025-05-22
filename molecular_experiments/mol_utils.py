import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from positional_embeddings import PositionalEmbedding

import sys
sys.path.append('..')
from curves import CubicSpline

### Energy approximation model tools

def mod_pi(x):
    """Modulo pi. This is required beause the energy function is defined on a torus"""
    y = x + torch.pi * torch.ones_like(x)
    y = torch.remainder(y, 2 * torch.pi)
    return y - torch.pi * torch.ones_like(x)

def extended_mod(x, c=0.5 * torch.pi):
    """"
    `Extended` modulo pi - enables additional wiggle room. Helpful for training the
    diffusion model
    """
    x = torch.where(x > torch.pi + c, x - 2 * torch.pi, x)
    x = torch.where(x < -torch.pi - c, x + 2 * torch.pi, x)
    return x


class PotentialModel(nn.Module):
    """Implementation of a fully connected neural network used to approximate the potential function"""
    def __init__(self, hidden_sizes):
        super().__init__()
        dims = [(2, hidden_sizes[0])] + [
            (hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(len(hidden_sizes) - 1)
        ]
        self.layers = [
            nn.Sequential(
                nn.Linear(dim[0], dim[1]),
                nn.SiLU()
            )
            for dim in dims
        ] + [nn.Linear(hidden_sizes[-1], 1)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

### Diffusion model tools

class Block(nn.Module):
    """Implementation from: https://github.com/tanelp/tiny-diffusion"""
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    """Implementation from: https://github.com/tanelp/tiny-diffusion"""
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


def alpha_sigma(l):
    """Calculates the SDE parameters for the VP-SDE parametrized with l=logSNR(t)"""
    alpha = torch.sigmoid(l).sqrt()
    sigma = torch.sigmoid(-l).sqrt()
    return alpha, sigma

### Spacetime geometry tools

def eta(theta):
    """
    Implementation of the natural parameter - Eq 18 in the paper.
    Parameters
    ----------
    theta: torch.Tensor
        a batch of spacetime points of shape (N, 3), where the first column is the `time` component, and the remaining two are the `space` component
    Returns
    ----------
    eta_t : torch.Tensor
        `time` component of the natural parameter - tensor or shape (N,)
    eta_x : torch.Tensor
        `space` component of the natural parameter - tensor of shape (N, 2)
    """
    l, x = theta[:, 0], theta[:, 1:]
    alpha, sigma = alpha_sigma(l)
    eta_t = -0.5 * l.exp()
    eta_x = (alpha/sigma ** 2).unsqueeze(1) * x
    return eta_t, eta_x

def mu(theta, model):
    """
    Implementation of the expectation parameter - Eq 22 in the paper. Since our data distribution is 1D, the spacetime is 2D
    Parameters
    ----------
    theta: torch.Tensor
        a batch of spacetime points of shape (N, 3), where the first column is the `time` component, and the remaining two are the `space` component
    Returns
    ----------
    mu_t : torch.Tensor
        `time` component of the expectation parameter - tensor or shape (N,)
    mu_x : torch.Tensor
        `space` component of the expectation parameter - tensor of shape (N, 2)
    """
    l, x = theta[:, 0], theta[:, 1:]
    x.requires_grad_(True)
    eds = model(x, l)
    div = 0
    div += torch.autograd.grad(eds[:, 0].sum(), x, create_graph=True)[0][:, 0]
    div += torch.autograd.grad(eds[:, 1].sum(), x, create_graph=True)[0][:, 1]
    alpha, sigma = alpha_sigma(l)
    mu_t = sigma ** 2/alpha * div + (eds ** 2).sum(dim=1)
    mu_x = eds
    return mu_t, mu_x

def energy_from_mu_eta(mu_t, mu_x, eta_t, eta_x):
    """
    Implementation of the energy of a discretized curve - Eq 23 in the paper.
    Parameters
    ----------
    mu_t : torch.Tensor
        `time` component of the expectation parameter, of shape (N, )
    mu_x : torch.Tensor
        `space` component of the expectation parameter, of shape (N, 2)
    eta_t : torch.Tensor
        `time` component of the natural parameter, of shape (N, )
    eta_x : torch.Tensor
        `space` component of the natural parameter, of shape (N, 2)
    Returns
    ----------
    energy : torch.Tensor
        Energy represented as tensor of shape (,)
    """
    # In theory, both t_prod and x_prod are always non-negative (because the Fisher-Rao metric tensor is SPD)
    # However, in practice we additionally enforce it with relu to avoid numerical instabilities
    t_prod = torch.relu((mu_t[1:] - mu_t[:-1]) * (eta_t[1:] - eta_t[:-1]))
    x_prod = torch.relu((mu_x[1:] - mu_x[:-1]) * (eta_x[1:] - eta_x[:-1])).sum(dim=1)
    energies = t_prod + x_prod
    return energies.sum()

def energy(theta, model, *args, **kwargs):
    assert theta.ndim == 2
    assert theta.shape[1] == 3
    mu_t, mu_x = mu(theta, model)
    eta_t, eta_x = eta(theta)
    return energy_from_mu_eta(mu_t, mu_x, eta_t, eta_x)

def rev_kl_from_mu_eta(mu_t, mu_x, eta_t, eta_x, theta_star):
    """
    Reverse KL implemented as top row of Equation 44 in the paper.
    Parameters
    ----------
    mu_t : torch.Tensor
        `time` component of the expectation parameter, of shape (N, )
    mu_x : torch.Tensor
        `space` component of the expectation parameter, of shape (N, 2)
    eta_t : torch.Tensor
        `time` component of the natural parameter, of shape (N, )
    eta_x : torch.Tensor
        `space` component of the natural parameter, of shape (N, 2)
    theta_star : torch.Tensor
        point in spacetime represented as tensor of shape (3,)
    Returns
    -----------
    rev_kl : torch.Tensor
        Reverse KL along a curve - tensor of shape (N - 1,)
    """
    mu_diff_t = mu_t[1:] - mu_t[:-1]
    mu_diff_x = mu_x[1:] - mu_x[:-1]
    eta_star_t, eta_star_x = eta(theta_star.unsqueeze(0))
    eta_star_t_diff = (eta_star_t - eta_star_t)[1:]
    eta_star_x_diff = (eta_star_x - eta_star_x)[1:]
    rev_kl = eta_star_t_diff * mu_diff_t + (eta_star_x_diff * mu_diff_x).sum(dim=1)
    rev_kl = torch.cumsum(rev_kl, dim=0)
    return rev_kl

def fwd_kl_from_mu_eta(mu_t, mu_x, eta_t, eta_x, model, theta_star):
    """
    Forward KL implemented as bottom row of Equation 44 in the paper.
    Parameters
    ----------
    mu_t : torch.Tensor
        `time` component of the expectation parameter, of shape (N, )
    mu_x : torch.Tensor
        `space` component of the expectation parameter, of shape (N, 2)
    eta_t : torch.Tensor
        `time` component of the natural parameter, of shape (N, )
    eta_x : torch.Tensor
        `space` component of the natural parameter, of shape (N, 2)
    theta_star : torch.Tensor
        point in spacetime represented as tensor of shape (3,)
    Returns
    -----------
    fwd_kl : torch.Tensor
        Forward KL along a curve - tensor of shape (N - 1,)
    """
    eta_diff_t = eta_t[1:] - eta_t[:-1]
    eta_diff_x = eta_x[1:] - eta_x[:-1]
    mu_star_t, mu_star_x = mu(theta_star.unsqueeze(0), model)
    mu_star_t_diff = (mu_t - mu_star_t)[1:]
    mu_star_x_diff = (mu_x - mu_star_x)[1:]
    fwd_kl = mu_star_t_diff * eta_diff_t + (mu_star_x_diff * eta_diff_x).sum(dim=1)
    fwd_kl = torch.cumsum(fwd_kl, dim=0)
    return fwd_kl

thresholding_fn = torch.nn.Softplus(beta=10)
def lambda_reg_from_step_num(step_num, thres_num):
    if step_num < thres_num:
        return 0.
    return min((step_num - thres_num) / 28, 100)

def low_variance_loss_fn(theta, model, step_num=None):
    mu_t, mu_x = mu(theta, model)
    eta_t, eta_x = eta(theta)
    energy = energy_from_mu_eta(mu_t, mu_x, eta_t, eta_x)
    penalty = thresholding_fn(3 - theta[:, 0]).mean()
    lambda_reg = lambda_reg_from_step_num(step_num, 1200)
    return energy + lambda_reg * penalty

def region_avoiding_loss_fn(theta, model, theta_star=None, step_num=None):
    mu_t, mu_x = mu(theta, model)
    eta_t, eta_x = eta(theta)
    energy = energy_from_mu_eta(mu_t, mu_x, eta_t, eta_x)
    lambda_reg = lambda_reg_from_step_num(step_num, 1200)
    low_variance_loss = thresholding_fn(3.75 - theta[:, 0]).mean()
    fwd_kl = fwd_kl_from_mu_eta(mu_t, mu_x, eta_t, eta_x, model, theta_star)
    fwd_kl_penalty = thresholding_fn(-4350 - fwd_kl).mean()
    loss = energy + lambda_reg * low_variance_loss + fwd_kl_penalty
    return loss

LOSS_FUNCTIONS = {
    'unconstrained': energy,
    'low-variance': low_variance_loss_fn,
    'region-avoiding': region_avoiding_loss_fn
}

def geodesic(theta1, theta2, model, n_opt_steps, num_intermediate_points, num_nodes=2, mode=None, **loss_fn_kwargs):
    """
    Implementation of approximate geodesic, parametrizes the curve as a CubicSpline and minimizes its energy w.r.t. curve's parameters using Adam optimizer
    Parameters
    ----------
    theta1 : torch.Tensor
        First endpoint of the curve, represented as a point in spacetime, i.e. (3,) tensor
    theta2 : torch.Tensor
        Second endpoint of the curve, represented as a point in spacetime, i.e. (3,) tensor
    model : torch.nn.Module
        Approximate denoiser
    n_opt_steps : int
        Number of optimization steps
    num_intermediate_points : int
        Number of points to discretize the curve into (in the paper: `N`)
    num_nodes : int
        Parameter of the CubicSpline. The higher the number of nodes, the more flexible/expressive the curve
    mode : str
        Which optimization mode. Possible options:
            'unconstrained' - minimize energy
            'low-variance' - minimize energy and penalize large variance of p(x|theta)
            'region-avoiding' - minimize energy and penalize being close to a restricted region
    """
    assert mode in ['unconstrained', 'low-variance', 'region-avoiding'], "'mode' must be one of 'unconstrained', 'low-variance', 'region-avoiding'."
    loss_fn = LOSS_FUNCTIONS[mode]
    curve = CubicSpline(begin=theta1, end=theta2, num_nodes=num_nodes)
    optimizer = torch.optim.Adam(curve.parameters(), lr=1e-1)
    t_tensor = torch.linspace(0, 1, num_intermediate_points).unsqueeze(0)
    pbar = tqdm.tqdm(range(n_opt_steps))
    for step_id in pbar:
        optimizer.zero_grad()
        theta = curve(t_tensor)[0]
        loss = loss_fn(theta, model, step_num=step_id, **loss_fn_kwargs)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"energy": loss.item()})
    return curve(t_tensor)[0].detach().numpy()


def denoising_energy(inp, theta, potential_model):
    """
    The energy of the denoising distribution implemented as in Eq 58 in the paper
    Parameters
    ----------
    theta : torch.Tensor
        point in Spacetime defining the denoising distribution, tensor of shape (3,)
    inp : torch.Tensor
        points in the data space for which we want to evaluate the energy, tensor of shape (N, 2)
    
    """
    assert theta.shape == (3,)
    l, x = theta[0], theta[1:].unsqueeze(0)
    alpha, sigma = alpha_sigma(l)
    z = potential_model(mod_pi(inp).float())[:, 0]
    resid = 0.5 * ((alpha * inp - x) ** 2).sum(dim=1) / sigma**2
    z += resid
    return z


def annealed_langevin_dynamics(potential_model, n_paths, n_lan_steps, ds, curve):
    """
    Implementation of annealed Langevin dynamics - as described in Algorithm 1 in the paper.
    Parameters
    ----------
    n_paths : int
        Number of transition paths to return
    n_lan_steps : int
        Number of steps of Langevin dynamics (in the paper: `K`)
    ds : float
        Step size in the Langevin dynamics simulation
    curve : np.ndarray
        Discretized spacetime curve of shape (N, 3), where the first column corresponds to the `time` component, and the remaining two to the `space` component
    Returns
    ----------
    transition_paths : np.ndarray
        Estimated transition paths - np.ndarray of shape (n_lan_steps * N, n_paths, 2)
    """
    ds = torch.tensor(ds)
    path = [np.tile(curve[0][1:], (n_paths, 1))] # initialize the transition paths with gamma_0
    curr_x = torch.from_numpy(path[0])
    for theta in tqdm.tqdm(torch.from_numpy(curve)): # outer loop in Algorithm 1
        for i in range(n_lan_steps): # inner loop in Algorithm 1
            curr_x_grad = curr_x.detach().clone().requires_grad_(True)
            pot_pred = denoising_energy(curr_x_grad, theta, potential_model)
            drift = -torch.autograd.grad(pot_pred.sum(), curr_x_grad)[0]
            curr_x = curr_x + 0.5 * ds * drift + torch.sqrt(ds) * torch.randn_like(curr_x) # Langevin step
            path.append(curr_x.detach().numpy()) # update path
    return np.array(path)


### Visualization tools

def visualize_energy(fig, ax, potential_model, colorbar=True, alpha=1):
    xs = np.arange(-np.pi, np.pi + .1, .1)
    ys = np.arange(-np.pi, np.pi + .1, .1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor([x, y]).view(2, -1).T
    z = potential_model(inp.float()).detach()
    z = z.view(y.shape[0], y.shape[1])
    cm = ax.contourf(xs, ys, z, levels=50, zorder=0, alpha=alpha)
    if colorbar:
        fig.colorbar(cm)


def visualize_spacetime_curve(curve, potential_model):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title(r'$U(\mathbf{x})$', fontsize=14)
    visualize_energy(fig, ax1, potential_model, alpha=1)
    theta1 = curve[0]
    theta2 = curve[-1]
    n_points = curve.shape[0]
    ax1.scatter(theta1[1], theta1[2], color='white', marker='x')
    ax1.scatter(theta2[1], theta2[2], color='white', marker='x')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.plot(curve[:, 1], curve[:, 2], color='C3')
    ax2.plot(np.linspace(0, 1, n_points), np.exp(-0.5 * curve[:, 0]))
    ax2.set_title(r'$\mathrm{SNR}(t(s))^{-1/2}$')

def visualize_restricted_region(ax, potential_model, theta_star):
    dx = 0.05
    constr_xs = np.arange(-np.pi, np.pi + dx, dx)
    constr_ys = np.arange(-np.pi, np.pi + dx, dx)
    constr_x, constr_y = np.meshgrid(constr_xs, constr_ys)
    constr_inp = torch.tensor([constr_x, constr_y]).view(2, -1).T
    constr_z = denoising_energy(constr_inp, theta_star, potential_model)
    constr_z = constr_z.view(constr_y.shape[0], constr_y.shape[1]).detach().numpy()
    constr_z = constr_z - np.min(constr_z)
    contour_plot = ax.contour(constr_xs, constr_ys, constr_z, levels=[15])
    handles_unfilled, labels = contour_plot.legend_elements()
    ax.legend(handles_unfilled, ["Restricted region"], loc='lower right')
    

def visualize_transition_paths(paths, potential_model, show_restricted_region=False, theta_star=None):
    start_step = 42000
    end_step = 5000
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4), width_ratios=[5, 4])

    ax1.set_title('Boltzmann distribution', fontsize=14)
    visualize_energy(fig, ax1, potential_model, alpha=0.8)
    theta1 = paths[0][0]
    theta2 = paths[-1][0]
    ax1.scatter(theta1[0], theta1[1], color='white', marker='x', label='low energy states', zorder=5)
    ax1.scatter(theta2[0], theta2[1], color='white', marker='x', zorder=5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    legend = ax1.legend(
        facecolor='dimgray',     # Background color
        framealpha=0.6,          # Optional: adds transparency
        edgecolor='dimgray', fontsize=11        # Optional: border color
    )

    # Change legend text color
    for text in legend.get_texts():
        text.set_color("white")  # Replace "red" with any color you want


    ax2.set_title(r'Transition paths', fontsize=14)# + r',  $\sqrt{\mathrm{SNR}(t(s))}$' + fr'$={t:.2f}$')
    visualize_energy(fig, ax2, potential_model, alpha=0.8, colorbar=False)
    ax2.scatter(theta1[0], theta1[1], color='white', marker='x', zorder=5)
    ax2.scatter(theta2[0], theta2[1], color='white', marker='x', zorder=5)
    for col_id, path_id in zip([0, 1, 3], [0, 1, 4]):
        ax2.plot(
            paths[:, path_id, 0][start_step:-end_step],
            paths[:, path_id, 1][start_step:-end_step],
            linewidth=0.5, color=f'C{col_id}')
    ax2.set_xlim((-np.pi, np.pi))
    ax2.set_ylim((-np.pi, np.pi))
    ax2.set_xticks([])
    ax2.set_yticks([])
    if show_restricted_region:
        visualize_restricted_region(ax2, potential_model, theta_star)
    plt.tight_layout()