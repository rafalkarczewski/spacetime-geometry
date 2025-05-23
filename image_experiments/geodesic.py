# This code is based on https://github.com/NVlabs/edm2/blob/main/generate_images.py

"""Generate a spacetime geodesic using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

from curves import CubicSpline

# torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),  # fid = 3.53
    'edm2-img512-s-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),   # fid = 2.56
    'edm2-img512-m-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),   # fid = 2.25
    'edm2-img512-l-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),   # fid = 2.06
    'edm2-img512-xl-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),  # fid = 1.96
    'edm2-img512-xxl-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'), # fid = 1.91
    'edm2-img64-s-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),    # fid = 1.58
    'edm2-img64-m-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),    # fid = 1.43
    'edm2-img64-l-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),    # fid = 1.33
    'edm2-img64-xl-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),   # fid = 1.33
    'edm2-img512-xs-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),  # fd_dinov2 = 103.39
    'edm2-img512-s-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),   # fd_dinov2 = 68.64
    'edm2-img512-m-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),   # fd_dinov2 = 58.44
    'edm2-img512-l-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),   # fd_dinov2 = 52.25
    'edm2-img512-xl-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),  # fd_dinov2 = 45.96
    'edm2-img512-xxl-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'), # fd_dinov2 = 42.84
    'edm2-img512-xs-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.4), # fid = 2.91
    'edm2-img512-s-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.4), # fid = 2.23
    'edm2-img512-m-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.2), # fid = 2.01
    'edm2-img512-l-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.88
    'edm2-img512-xl-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.2), # fid = 1.85
    'edm2-img512-xxl-guid-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.81
    'edm2-img512-xs-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.7), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.9), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.0), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',    gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.7), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',   gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.7), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.7), # fd_dinov2 = 33.09
}

#----------------------------------------------------------------------------

def edm_encoder(
    net, lat, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    """
    EDM encoder which takes a latent code and simulates the PF-ODE from t=sigma_min to t=sigma_max using the 2nd order Heun solver proposed by Karras et al. "Elucidating the Design space of Diffusion Models" (NeurIPS 2022). The entire trajectory is saved for plotting purposes
    Parameters taken exactly as in the original implementation: https://github.com/NVlabs/edm2/blob/main/generate_images.py
    Returns:
    -----------
    trajectory : torch.Tensor
        returns the entire trajectory as a spacetime curve (where instead of using time `t` parametrization, we use `l=logSNR(t)`)
    """
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=lat.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    t_steps = torch.flip(t_steps, dims=(0,))

    # Main sampling loop.
    x_next = lat.to(dtype)
    traj = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        lambda_cur = -2 * torch.log(t_cur)
        lambda_cur = lambda_cur.unsqueeze(0)
        curr_point = torch.cat([lambda_cur, x_cur.flatten()])
        traj.append(curr_point)
        d_cur = (x_cur - denoise(x_cur, t_cur)) / t_cur

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            x_next = x_cur + (t_next - t_cur) * d_cur * 2 + (t_cur ** 2 - t_next ** 2).sqrt() * randn_like(x_cur)
        else:
            x_next = x_cur + (t_next - t_cur) * d_cur
            # Apply 2nd order correction.
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    
    lambda_cur = -2 * torch.log(t_next)
    lambda_cur = lambda_cur.unsqueeze(0)
    curr_point = torch.cat([lambda_cur, x_next.flatten()])
    traj.append(curr_point)

    return torch.stack(traj)

def mu(denoise_fn, theta, eps):
    """
    The expectation parameter as defined in Eq22 in the paper.
    Parameters
    ----------
    denoise_fn : Callable
        implementation of the denoiser x0_hat(xt, t)
    theta : torch.Tensor
        Curve discretized into N points represented as (N, D+1) shape tensor. First column is the `time` component, and the remaining D columns is the `space` component
    eps : torch.Tensor
        Rademacher noise tensor as (N, D) tensor - required to estimate divergence with Hutchinson's trick
    Returns
    ----------
    mu_l : torch.Tensor
        `time` component of the expectation parameter represented as (N,) tensor
    mu_x : torch.Tensor
        `space` component of the expectation paramter represented as (N, D) tensor
    """
    l, x = theta[:, 0], theta[:, 1:]
    x_img = x.reshape((-1, 4, 64, 64))
    eps_img = eps.reshape((-1, 4, 64, 64))
    def f(l, x): # redefine the denoiser, which uses time `t` as input, whereas we are working with l=logSNR(t) parametrization
        t = torch.exp(-0.5 * l)  # t = logSNR^(-1)(l)
        return denoise_fn(x, t)
    with torch.enable_grad():
        x_img.requires_grad_(True)
        mu_x = f(l, x_img)
        mu_x_eps = torch.sum(mu_x * eps_img)
        mu_x_eps_grad = torch.autograd.grad(mu_x_eps, x_img, create_graph=True)[0]
        mu_div = torch.sum(eps_img * mu_x_eps_grad, dim=tuple(range(1, len(x_img.shape)))) # divergence estimated with Hutchinson's trick
        mu_l = torch.exp(-l) * mu_div + torch.sum(mu_x ** 2, dim=tuple(range(1, len(x_img.shape))))
        return mu_l, mu_x

def eta(theta):
    """
    Implementation of the natural parameters as defined in Eq18 in the paper
    Parameters
    ----------
    theta : torch.Tensor
        Curve discretized into N points represented as (N, D+1) shape tensor. First column is the `time` component, and the remaining D columns is the `space` component
    Returns
    ----------
    eta_l : torch.Tensor
        `time` component of the natural parameter represented as (N,) tensor
    eta_x : torch.Tensor
        `space` component of the natural paramter represented as (N, D) tensor
    """
    l, x = theta[:, 0], theta[:, 1:]
    return -0.5 * torch.exp(l), torch.exp(l)[:, None, None, None] * x.reshape((-1, 4, 64, 64))

def compute_batch_energy(denoise_fn, points, randint=torch.randint):
    """
    Implementation of the energy of a discretized curve as defined in Eq23 in the paper.
    Parameters
    ----------
    denoise_fn : Callable
        implementation of the denoiser x0_hat(xt, t)
    points : torch.Tensor
        Curve discretized into N points represented as (N, D+1) shape tensor. First column is the `time` component, and the remaining D columns is the `space` component
    randint : Callable
        function that generation random integer tensors. Required to generate Rademacher variables
    Returns
    ----------
    energy: torch.tensor
        value of the energy as a differentiable tensor
    """
    eta_t, eta_x = eta(points)
    all_eps = randint(low=0, high=2, size=(1, points.shape[0], 4 * 64 * 64), dtype=points.dtype, device=points.device) * 2 - 1
    mu_t, mu_x = mu(denoise_fn, points, all_eps[0])
    t_prod = (eta_t[1:] - eta_t[:-1]) * (mu_t[1:] - mu_t[:-1])
    x_prod = (eta_x[1:] - eta_x[:-1]) * (mu_x[1:] - mu_x[:-1])
    
    # Both `t_prod` and `x_prod` are always provably non-negative (because the Fisher-Rao metric is always (semi-)positive definite), but in practice, we additionally enforce it with relu to avoid numerical instabilities.

    t_prod = torch.relu(t_prod)
    x_prod = torch.relu(x_prod)

    x_prod = torch.sum(x_prod, dim=tuple(range(1, len(eta_x.shape))))
    return torch.sum(t_prod + x_prod)

def compute_energy(denoise_fn, geodesic, n_points, device, batch_size, randint):
    """
    Implementation of the energy function by splitting the curve into chunks that fit in the GPU. This function estimates the energy of the curve and performs the backward step on the parameters of the curve.
    Parameters
    ----------
    denoise_fn : Callable
        implementation of the denoiser x0_hat(xt, t)
    geodesic : utils.CubicSpline
        Curve represented as a cubic spine.
    n_points : int
        Number of points (In the paper: N) that we discretize the curve into
    device : torch.device
        Whether we run on CPU or GPU (we never tested this model on CPU)
    batch_size: int
        The size each chunk that we split the curve into
    randint : Callable
        function that generation random integer tensors. Required to generate Rademacher variables
    Returns
    ----------
    energy: torch.float
        value of the energy as a float
    """
    energy = 0
    t_tensor = torch.linspace(0, 1, n_points, device=device).unsqueeze(0)
    for i in range(0, n_points - 1, batch_size - 1): # The loop runs in a way that the last element of i-th batch needs to appear again as the first element of the (i+1)-th batch. This is to ensure that the energy over the entire curve is properly estimated. The boundary points need to be evaluated twice.
        points = geodesic(t_tensor)[0]
        energy_el = compute_batch_energy(denoise_fn, points[i:i + batch_size], randint=randint)
        energy_el = (n_points - 1) * energy_el # (n_points - 1) = 1 / ds
        if torch.isfinite(energy_el): # This is to ensure ignoring batches for which the energy value explodes. Useful in initial experiments - not used in final experiments
            energy_el.backward() # on the each chunk we call .backward(), instead of taking the sum over the entire curve first, to release the computational graph and avoid memory issues
            energy += energy_el.item() # for logging purposes
        else:
            del energy_el
    return energy

def geodesic_optimization_step(optimizer, denoise_fn, geodesic, n_points, device, batch_size, randint):
    """
    A single optimization step of the energy with respect to the curve's parameters
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    denoise_fn : Callable
        implementation of the denoiser x0_hat(xt, t)
    geodesic : utils.CubicSpline
        Curve represented as a cubic spine.
    n_points : int
        Number of points (In the paper: N) that we discretize the curve into
    device : torch.device
        Whether we run on CPU or GPU (we never tested this model on CPU)
    batch_size: int
        The size each chunk that we split the curve into
    randint : Callable
        function that generation random integer tensors. Required to generate Rademacher variables
    Returns
    ----------
    None
    """
    optimizer.zero_grad()
    energy = compute_energy(denoise_fn, geodesic, n_points, device, batch_size, randint)
    param_size = (geodesic.params.data ** 2).sum().sqrt().item()
    grad_norm = (geodesic.params.grad ** 2).sum().sqrt().item()
    print(f'Energy: {energy:.2f}.Geodesic param size: {param_size:.2f}. Grad norm: {grad_norm:.2f}', flush=True)
    optimizer.step()

def optimize_geodesic(geodesic, denoise_fn, n_opt_steps_warmup, n_opt_steps_finetune, batch_size, randint):
    """
    Optimization of the energy with respect to the parameters of the curve in two phases: Initial warmup phase with the curve discretized into 16 points, and second finetuning phase with discretizing the curve into 64 points
    Parameters
    ----------
    geodesic : utils.CubicSpline
        Curve represented as a cubic spine.
    denoise_fn : Callable
        implementation of the denoiser x0_hat(xt, t)
    n_steps_warmup : int
        The number of optimization steps in the initial warmup phase
    n_opt_steps_finetune : int
        The number of finetuning steps
    batch_size: int
        The size each chunk that we split the curve into
    randint : Callable
        function that generation random integer tensors. Required to generate Rademacher variables
    Returns
    ----------
    optimized_curve : torch.Tensor
        curve after optimization represented as (64, D + 1) dimensional tensor
    params : torch.state_dict
        parameters of the curve after optimization
    """
    optimizer = torch.optim.AdamW(geodesic.parameters(), lr=1e-1)
    for _ in tqdm.tqdm(range(n_opt_steps_warmup), unit='opt step'):
        geodesic_optimization_step(optimizer, denoise_fn, geodesic, n_points=16,
        device=geodesic.device, batch_size=batch_size, randint=randint)
    for _ in tqdm.tqdm(range(n_opt_steps_finetune), unit='opt step'):
        geodesic_optimization_step(optimizer, denoise_fn, geodesic, n_points=64,
        device=geodesic.device, batch_size=batch_size, randint=randint)
    return geodesic(torch.linspace(0, 1, 64, device=geodesic.device).unsqueeze(0))[0], geodesic.state_dict()

def rescale_noisy_points(path, data_std=0.5):
    """
    Rescaling of noisy points before feeding them to VAE decoder.
    This is to avoid unrealistic pixel values
    """
    l_points, x_noisy_points = path[:, 0], path[:, 1:]
    s2 = (-l_points).exp()
    normalizing_factor = data_std / (data_std ** 2 + s2).sqrt()
    return (normalizing_factor.unsqueeze(1) * x_noisy_points).reshape((-1, 4, 64, 64))

def geodesic(net, labels, gnet, lat1, lat2, num_nodes, n_opt_steps_warmup, n_opt_steps_finetune,
    batch_size, randint=torch.randint, randn_like=torch.randn_like,
    guidance=None, dtype=None):
    """
    Main function performing spacetime interpolation between clean images. 
    Parameters
    ----------
    net : torch.nn.Module
        The denoiser network
    labels : torch.Tensor
        one-hot encoded label to use as the conditioning of the denoiser model
    gnet : torch.nn.Module
        guidance model
    lat1 : torch.Tensor
        latent code of the first endpoint of interpolation
    lat2 : torch.Tensor
        latent code of the second endpoint of interpolation
    num_nodes : int
        Number of nodes of the cubic spline. The more points the more flexible the curve
    n_opt_steps_warmup : int
        Number of the initial warmup phase of optimization. In this phase, the curve is discretized into 16 points
    n_opt_steps_finetune : int
        Number of the finetuning phase of optimization. In this phase, the curve is discretized into 64 points
    batch_size : int
        Size of each chunk that the curve is split into. We recommend setting as high as the GPU's memory allows
    Returns
    ----------
    lambdas : torch.Tensor
        the `time` component of the optimized geodesic represented as `l=logSNR(t)` instead of `t`. This is a tensor of shape (64,)
    noisy_points : torch.Tensor
        the `space` component of the optimized geodesic represented as a (N, 4, 64, 64) tensor (N noisy images)
    denoised_points : torch.Tensor
        The expected denoising of each point on the optimized geodesic represented as (N, 4, 64, 64) tensor (N denoised images)
    params : torch.state_dict
        Parameters of the optimized geodesic
    """
    def denoise_fn(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)
        
    SIGMA_MIN = 0.36787944117144233 # Noise level chosen to avoid numerical instabilities for low values of t. This value of t corresponds to logSNR(t)=2, which we found yields almost imperceptible variations in the denoising distribution

    encoder_fn = lambda latent_code: edm_encoder(net, latent_code, labels=labels, gnet=gnet, num_steps=32, sigma_min=0.002, sigma_max=SIGMA_MIN, rho=7., guidance=guidance,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, dtype=dtype, randn_like=randn_like)

    # compute the encoding trajectories for the latent codes with PF-ODE
    traj1 = encoder_fn(lat1)
    traj2 = encoder_fn(lat2)

    # Define endpoints for the curve to be optimized
    theta1 = traj1[-1]
    theta2 = traj2[-1]

    geodesic = CubicSpline(begin=theta1, end=theta2, num_nodes=num_nodes)
    geodesic, params = optimize_geodesic(geodesic, denoise_fn,
        n_opt_steps_warmup, n_opt_steps_finetune, batch_size, randint)

    # Merge the optimized geodesic with the PF-ODE encoding trajectories for visualization purposes
    geodesic = torch.cat([
        traj1[:-1], geodesic, torch.flip(traj2[:-1], dims=(0,))
    ])

    l_points, x_noisy_points = geodesic[:, 0], geodesic[:, 1:]
    x_noisy_points = x_noisy_points.reshape((-1, 4, 64, 64))
    t_points = torch.exp(-0.5 * l_points)
    denoised_points = []
    for t, noisy_x in zip(t_points, x_noisy_points): # 
        denoised_points.append(denoise_fn(noisy_x.unsqueeze(0), t.unsqueeze(0)).squeeze(0).detach().cpu())
    denoised_points = torch.stack(denoised_points)
    return l_points, rescale_noisy_points(geodesic.detach().cpu()), denoised_points, params

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = geodesic,             # Which sampler function to use.
    img1                = None,
    img2                = None,
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        for p in net.parameters():
            p.requires_grad = False
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Optimizing a geodesic...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.labels = None
                    if net.label_dim > 0: # If tha class label is not provided, generate a random one
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Load images and transform them into tensors
                    img1_tensor = torch.from_numpy(np.array(PIL.Image.open(img1))).permute(2, 0, 1).to(torch.device('cuda')).unsqueeze(0)
                    img2_tensor = torch.from_numpy(np.array(PIL.Image.open(img2))).permute(2, 0, 1).to(torch.device('cuda')).unsqueeze(0)

                    # Encode raw images to (4, 64, 64) latent codes with StabilityVAE encoder
                    img1_latent = encoder.encode_latents(encoder.encode_pixels(img1_tensor))
                    img2_latent = encoder.encode_latents(encoder.encode_pixels(img2_tensor))

                    # Estimate the geodesic between latent codes
                    l_points, noisy_latents, denoised_latents, geodesic_params = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net,
                        labels=r.labels, gnet=gnet, lat1=img1_latent, lat2=img2_latent, randint=rnd.randint, randn_like=rnd.randn_like,
                        **sampler_kwargs)
                    
                    # Decode noisy points and estimated mean denoisings with StabilityVAE decoder
                    r.noisy_images = encoder.decode(noisy_latents.to(torch.device('cuda'))).detach().cpu()
                    r.denoised_images = encoder.decode(denoised_latents.to(torch.device('cuda'))).detach().cpu()

                    # Save images.
                    if outdir is not None:
                        os.makedirs(outdir, exist_ok=True)
                        torch.save(geodesic_params, os.path.join(outdir, 'geodesic_params.pt'))
                        for t_idx, (noisy_image, denoised_image) in enumerate(zip(
                            r.noisy_images.permute(0, 2, 3, 1).cpu().numpy(),
                            r.denoised_images.permute(0, 2, 3, 1).cpu().numpy()
                        )):
                            PIL.Image.fromarray(noisy_image, 'RGB').save(os.path.join(outdir, f'noisy_{t_idx}.png'))
                            PIL.Image.fromarray(denoised_image, 'RGB').save(os.path.join(outdir, f'denoised_{t_idx}.png'))
                            np.save(os.path.join(outdir, 'lambdas.npy'), l_points.cpu().detach().numpy())


                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--gnet',                     help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--num_nodes',                help='Num nodes of the cubic spline', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--n_opt_steps_warmup',       help='Number of initial geodesic opt steps', metavar='INT',                     type=click.IntRange(min=0), required=True)
@click.option('--n_opt_steps_finetune',     help='Number of finetuning geodesic opt steps', metavar='INT',                     type=click.IntRange(min=0), required=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--batch_size',               help='Batch size in geodesic eval', metavar='INT',                      type=click.IntRange(min=1), required=True)

@click.option('--img1',                     help='startpoint of geodesic', metavar='DIR',                           type=str, required=True)
@click.option('--img2',                     help='endpoint of geodesic', metavar='DIR',                             type=str, required=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
