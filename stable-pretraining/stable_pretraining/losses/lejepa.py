"""LeJEPA SSL losses.

This module provides the SIGReg loss and LeJEPA combined loss for
joint-embedding self-supervised learning.

Reference:
    Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable
    Self-Supervised Learning Without the Heuristics. arXiv:2511.08544
"""

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """Sketched Isotropic Gaussian Regularization (SIGReg) loss.

    SIGReg constrains embeddings to follow an isotropic Gaussian distribution
    by comparing empirical characteristic functions against the theoretical
    N(0,1) characteristic function using random slicing projections.

    This implementation uses the Epps-Pulley test with trapezoidal quadrature
    for efficient computation. The symmetric property of the characteristic
    function is leveraged by integrating on [0, t_max] and doubling.

    Args:
        num_slices: Number of random 1D projections for slicing. Default: 256.
        t_max: Maximum integration domain for characteristic function.
            Default: 3.0 (corresponds to [-3, 3] effective range).
        n_points: Number of quadrature points for trapezoidal rule. Default: 17.

    Example:
        >>> sigreg = SIGRegLoss(num_slices=256, t_max=3.0, n_points=17)
        >>> projections = torch.randn(8, 4, 128)  # [V, N, K]
        >>> loss = sigreg(projections)

    Note:
        The input should be normalized (zero mean, unit variance per dimension)
        for optimal performance. The backbone projector typically handles this.
    """

    def __init__(
        self,
        num_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.num_slices = num_slices

        # Trapezoidal quadrature setup
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)

        # Trapezoidal weights (endpoints get half weight)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # Gaussian window: theoretical CF of N(0,1) is exp(-t^2/2)
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: Projected embeddings of shape [V, N, K] where:
                V = number of views
                N = batch size
                K = projection dimension

        Returns:
            Scalar tensor: Mean SIGReg statistic across all views.
        """
        # Generate random unit vectors for slicing
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # Project onto random directions: [V, N, num_slices]
        # Then expand for quadrature: [V, N, num_slices, n_points]
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Empirical characteristic function vs theoretical N(0,1)
        # cos term: Re(ECF) should match phi = exp(-t^2/2)
        # sin term: Im(ECF) should be 0 for symmetric distributions
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # Weighted integration (quadrature) and scale by sample size
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()


class LeJEPALoss(nn.Module):
    """Combined LeJEPA loss: Prediction loss + SIGReg regularization.

    LeJEPA combines two objectives:
    1. Prediction/Invariance loss: Encourages views of the same image to have
       similar representations by minimizing MSE between each view's projection
       and the mean projection across views.
    2. SIGReg loss: Prevents representation collapse by constraining the
       embedding distribution to be isotropic Gaussian.

    The combined loss is: L = (1 - lambda) * L_pred + lambda * L_sigreg

    Args:
        lambda_: Weight for SIGReg loss. Prediction loss weight is (1 - lambda_).
            Default: 0.05 for ImageNet-scale, 0.02 for smaller datasets.
        num_slices: Number of random projections for SIGReg. Default: 256.
        t_max: Integration domain for SIGReg. Default: 3.0.
        n_points: Quadrature points for SIGReg. Default: 17.

    Example:
        >>> loss_fn = LeJEPALoss(lambda_=0.02)
        >>> projections = torch.randn(4, 256, 128)  # [V, N, K]
        >>> loss, pred_loss, sigreg_loss = loss_fn(projections, return_components=True)

    Reference:
        Algorithm 2 in the LeJEPA paper (arXiv:2511.08544)
    """

    def __init__(
        self,
        lambda_: float = 0.05,
        num_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.sigreg = SIGRegLoss(
            num_slices=num_slices,
            t_max=t_max,
            n_points=n_points,
        )

    def forward(
        self,
        proj: torch.Tensor,
        return_components: bool = False,
        global_view_count: int | None = None,
        global_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LeJEPA loss.

        Args:
            proj: Projected embeddings of shape [V, N, K] where:
                V = number of views
                N = batch size
                K = projection dimension
            return_components: If True, also return individual loss components.
            global_view_count: Number of global views to compute centers from.
                If provided, uses proj[:global_view_count] as centers source.
            global_mask: Boolean mask of shape [V] indicating global views.
                If provided, uses proj[global_mask] as centers source.

        Returns:
            If return_components=False:
                Combined LeJEPA loss (scalar tensor)
            If return_components=True:
                Tuple of (total_loss, prediction_loss, sigreg_loss)
        """
        if global_mask is not None and global_view_count is not None:
            raise ValueError("Provide only one of global_mask or global_view_count.")

        # Select global views for centers (Algorithm 2 in the paper)
        if global_mask is not None:
            global_proj = proj[global_mask]
        elif global_view_count is not None:
            global_proj = proj[:global_view_count]
        else:
            global_proj = proj

        if global_proj.numel() == 0:
            raise ValueError("No global views provided for center computation.")

        # Prediction/Invariance loss: MSE between each view and mean of global views
        # proj: [V, N, K], centers: [N, K]
        # (centers - proj): [V, N, K] broadcast subtraction
        centers = global_proj.mean(0)
        pred_loss = (centers - proj).square().mean()

        # SIGReg loss
        sigreg_loss = self.sigreg(proj)

        # Combined loss
        total_loss = (1 - self.lambda_) * pred_loss + self.lambda_ * sigreg_loss

        if return_components:
            return total_loss, pred_loss, sigreg_loss
        return total_loss
