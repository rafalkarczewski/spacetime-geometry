### Implementation based on https://github.com/MachineLearningLifeScience/meaningful-protein-representations/blob/master/models/geoml/curve.py

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from matplotlib.axis import Axis
from torch import nn


class BasicCurve(ABC, nn.Module):
    def __init__(
        self,
        begin: torch.Tensor,
        end: torch.Tensor,
        num_nodes: int = 5,
        requires_grad: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._num_nodes = num_nodes
        self._requires_grad = requires_grad

        # if either begin or end only has one point, while the other has a batch
        # then we expand the singular point. End result is that both begin and
        # end should have shape BxD
        batch_begin = 1 if len(begin.shape) == 1 else begin.shape[0]
        batch_end = 1 if len(end.shape) == 1 else end.shape[0]
        if batch_begin == 1 and batch_end == 1:
            _begin = begin.detach().view((1, -1))  # 1xD
            _end = end.detach().view((1, -1))  # 1xD
        elif batch_begin == 1:  # batch_end > 1
            _begin = begin.detach().view((1, -1)).repeat(batch_end, 1)  # BxD
            _end = end.detach()  # BxD
        elif batch_end == 1:  # batch_begin > 1
            _begin = begin.detach()  # BxD
            _end = end.detach().view((1, -1)).repeat(batch_begin, 1)  # BxD
        elif batch_begin == batch_end:
            _begin = begin.detach()  # BxD
            _end = end.detach()  # BxD
        else:
            raise ValueError("BasicCurve.__init__ requires begin and end points to have " "the same shape")

        # register begin and end as buffers
        self.register_buffer("begin", _begin)  # BxD
        self.register_buffer("end", _end)  # BxD

        # overriden by child modules
        self._init_params(*args, **kwargs)

    @abstractmethod
    def _init_params(self, *args, **kwargs) -> None:
        pass

class CubicSpline(BasicCurve):
    def __init__(
        self,
        begin: torch.Tensor,
        end: torch.Tensor,
        num_nodes: int = 5,
        requires_grad: bool = True,
        basis: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(begin, end, num_nodes, requires_grad, basis=basis, params=params)
        self.device = begin.device

    def _init_params(self, basis, params) -> None:
        if basis is None:
            basis = self._compute_basis(num_edges=self._num_nodes - 1).to(self.begin.device)
        self.register_buffer("basis", basis)

        if params is None:
            params = torch.zeros(
                self.begin.shape[0], self.basis.shape[1], self.begin.shape[1],
                dtype=self.begin.dtype, device=self.begin.device
            )
        else:
            params = params.unsqueeze(0) if params.ndim == 2 else params

        if self._requires_grad:
            self.register_parameter("params", nn.Parameter(params))
        else:
            self.register_buffer("params", params)

    # Compute cubic spline basis with end-points (0, 0) and (1, 0)
    def _compute_basis(self, num_edges) -> torch.Tensor:
        with torch.no_grad():
            # set up constraints
            t = torch.linspace(0, 1, num_edges + 1, dtype=self.begin.dtype)[1:-1]

            end_points = torch.zeros(2, 4 * num_edges, dtype=self.begin.dtype)
            end_points[0, 0] = 1.0
            end_points[1, -4:] = 1.0

            zeroth = torch.zeros(num_edges - 1, 4 * num_edges, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor([1.0, t[i], t[i] ** 2, t[i] ** 3], dtype=self.begin.dtype)
                zeroth[i, si : (si + 4)] = fill
                zeroth[i, (si + 4) : (si + 8)] = -fill

            first = torch.zeros(num_edges - 1, 4 * num_edges, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor([0.0, 1.0, 2.0 * t[i], 3.0 * t[i] ** 2], dtype=self.begin.dtype)
                first[i, si : (si + 4)] = fill
                first[i, (si + 4) : (si + 8)] = -fill

            second = torch.zeros(num_edges - 1, 4 * num_edges, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor([0.0, 0.0, 6.0 * t[i], 2.0], dtype=self.begin.dtype)
                second[i, si : (si + 4)] = fill
                second[i, (si + 4) : (si + 8)] = -fill

            constraints = torch.cat((end_points, zeroth, first, second))
            self.constraints = constraints

            # Compute null space, which forms our basis
            _, S, V = torch.svd(constraints, some=False)
            basis = V[:, S.numel() :]  # (num_coeffs)x(intr_dim)

            return basis

    def _get_coeffs(self) -> torch.Tensor:
        coeffs = (
            self.basis.unsqueeze(0).expand(self.params.shape[0], -1, -1).bmm(self.params)
        )  # Bx(num_coeffs)xD
        B, num_coeffs, D = coeffs.shape
        degree = 4
        num_edges = num_coeffs // degree
        coeffs = coeffs.view(B, num_edges, degree, D)  # Bx(num_edges)x4xD
        return coeffs

    def _eval_polynomials(self, t: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        # each row of coeffs should be of the form c0, c1, c2, ... representing polynomials
        # of the form c0 + c1*t + c2*t^2 + ...
        # coeffs: Bx(num_edges)x(degree)xD
        B, num_edges, degree, D = coeffs.shape
        idx = torch.floor(t * num_edges).clamp(min=0, max=num_edges - 1).long()    # B x |t|
        power = (
            torch.arange(0.0, degree, dtype=t.dtype, device=self.device)
            .view(1, 1, -1)
            .expand(B, -1, -1)
        )                                                                           # B x  1  x (degree)
        tpow = t.view(B, -1, 1).pow(power)                                          # B x |t| x (degree)
        coeffs_idx = torch.cat([coeffs[k, idx[k]].unsqueeze(0) for k in range(B)])  # B x |t| x (degree) x D
        retval = tpow.unsqueeze(-1).expand(-1, -1, -1, D) * coeffs_idx              # B x |t| x (degree) x D
        retval = torch.sum(retval, dim=2)                                           # B x |t| x D
        return retval

    def _eval_straight_line(self, t: torch.Tensor) -> torch.Tensor:
        B, T = t.shape
        tt = t.view(B, T, 1)               # B x |t| x 1
        begin = self.begin.unsqueeze(1)    # B x  1  x D
        end = self.end.unsqueeze(1)        # B x  1  x D
        return (end - begin) * tt + begin  # B x |t| x D

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        coeffs = self._get_coeffs()  # Bx(num_edges)x4xD
        no_batch = t.ndim == 1
        if no_batch:
            t = t.expand(coeffs.shape[0], -1)  # Bx|t|
        retval = self._eval_polynomials(t, coeffs)  # Bx|t|xD
        retval += self._eval_straight_line(t)
        if no_batch and retval.shape[0] == 1:
            retval = retval.squeeze(0)  # |t|xD
        return retval
