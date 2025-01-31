"""
Kermut without structure so that we can use it for indels.

:
  _target_: kermut.kernels.SequenceKernel
  kernel_type: RBF

use_structure_kernel: true

use_zero_shot: true
zero_shot_method: ESM2/650M

use_sequence_kernel: true
embedding_type: ESM2
embedding_dim: 1280

noise_prior_scale: 0.1
use_prior: true
"""
from typing import Literal, Optional, Tuple

import torch
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from tqdm import trange


class SequenceKernel(Kernel):
    """Wrapper class for sequence (i.e., embedding) kernels that implements either RBF or
    Matérn kernels.

    Args:
        kernel_type: Type of kernel to use. Must be either "RBF" or "Matern".
        nu: The smoothness parameter for the Matérn kernel. Required if kernel_type
            is "Matern". Must be one of [0.5, 1.5, 2.5].
        **kwargs: Additional keyword arguments to be passed to the underlying kernel
            implementation.

    Raises:
        NotImplementedError: If kernel_type is not "RBF" or "Matern".
        AssertionError: If kernel_type is "Matern" and nu is not in [0.5, 1.5, 2.5].

    Attributes:
        kernel_type: The type of kernel being used ("RBF" or "Matern").
        nu: The smoothness parameter for Matérn kernel (None for RBF).
        base_kernel: The underlying kernel implementation (RBFKernel or MaternKernel).
    """

    def __init__(
        self,
        kernel_type: Literal["RBF", "Matern"] =  'RBF',
        nu: Optional[float] = None,
        **kwargs,
    ):
        super(SequenceKernel, self).__init__(**kwargs)
        self.kernel_type = kernel_type
        self.nu = nu
        self.base_kernel = self._create_kernel(kernel_type, nu, **kwargs)

    def _create_kernel(self, kernel_type, nu: Optional[float] = None, **params):
        if kernel_type == "RBF":
            return RBFKernel(**params)
        elif kernel_type == "Matern":
            assert nu in [0.5, 1.5, 2.5]
            return MaternKernel(nu=nu, **params)
        else:
            raise NotImplementedError

    def forward(self, x1, x2, diag=False, **params):
        return self.base_kernel.forward(x1, x2, diag=diag, **params)

    @property
    def is_stationary(self) -> bool:
        return True
    


class KermutGP(ExactGP):
    """Gaussian Process regression model for supervised variant effects predictions.

    A specialized Gaussian Process implementation that combines sequence and structural
    information for predicting the effects of protein mutations. It extends gpytorch's
    ExactGP class and supports both composite and single kernel architectures, as well
    as zero-shot prediction capabilities through its mean function.

    Args:
        train_inputs: Training input data for the GP model. Default expects tuple of
            (one-hot sequences, sequence_embeddings, zero-shot scores).
        train_targets: Target values corresponding to the training inputs.
        likelihood: Gaussian likelihood function for the GP model.
        kernel_cfg (DictConfig): Configuration dictionary for kernel specifications,
            containing settings for sequence_kernel and structure_kernel if composite
            is True, or a single kernel configuration if composite is False.
        use_zero_shot_mean (bool, optional): Whether to use a linear mean function
            for zero-shot predictions. If True, uses LinearMean; if False, uses
            ConstantMean. Defaults to True.
        composite (bool, optional): Whether to use a composite kernel combining
            sequence and structure information. If False, uses a single kernel
            specified in kernel_cfg. Defaults to True.
        **kwargs: Additional keyword arguments passed to the kernel initialization.

    Attributes:
        covar_module: The kernel (covariance) function, either a CompositeKernel
            or a single kernel as specified by kernel_cfg.
        mean_module: The mean function, either LinearMean for zero-shot predictions
            or ConstantMean for standard GP regression.
        use_zero_shot_mean (bool): Flag indicating whether zero-shot mean function
            is being used.
    """

    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        # kernel_cfg: DictConfig,
        use_zero_shot_mean: bool = True,
        composite: bool = True,
        **kwargs,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        # if composite:
        #     self.covar_module = CompositeKernel(
        #         sequence_kernel=kernel_cfg.sequence_kernel,
        #         structure_kernel=kernel_cfg.structure_kernel,
        #         **kwargs,
        #     )
        # else:
            # self.covar_module = hydra.utils.instantiate(kernel_cfg.kernel, **kwargs)
        self.covar_module = SequenceKernel(kernel_type='RBF', nu=None)

        self.use_zero_shot_mean = use_zero_shot_mean
        if self.use_zero_shot_mean:
            self.mean_module = LinearMean(input_size=1, bias=True)
        else:
            self.mean_module = ConstantMean()

    def forward(self, x_embed, x_zero=None) -> MultivariateNormal:
        # if x_zero is None:
            # x_zero = x_toks
        mean_x = self.mean_module(x_zero)
        covar_x = self.covar_module(x_embed)
        return MultivariateNormal(mean_x, covar_x)
    

def optimize_gp(
    gp: ExactGP,
    likelihood: GaussianLikelihood,
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    lr: float = 3.0e-4,
    n_steps: int = 150,
    progress_bar: bool = True,
) -> Tuple[ExactGP, GaussianLikelihood]:
    """Optimizes a Gaussian Process using marginal likelihood maximization.

    Trains the GP model by minimizing the negative log marginal likelihood using
    the AdamW optimizer. The function handles training mode activation, optimizer
    configuration, and iterative optimization with optional progress bar display.

    Args:
        gp: The Gaussian Process model to be optimized. Must be an instance
            of ExactGP.
        likelihood: The Gaussian likelihood function associated with the GP model.
        train_inputs: Tuple of input tensors for training. None values in the tuple
            will be filtered out.
        train_targets: Target values tensor for training.
        lr: Learning rate for the AdamW optimizer. Default is 3.0e-4.
        n_steps: Number of optimization steps. Default is 150.
        progress_bar: Boolean controlling progress bar display. Default is True.

    Returns:
        A tuple containing:
            - ExactGP: The optimized Gaussian Process model
            - GaussianLikelihood: The optimized likelihood function

    Note:
        The function uses ExactMarginalLogLikelihood as the loss function and
        optimizes model parameters using gradient descent. The progress bar
        can be toggled through the configuration.
    """
    gp.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, gp)

    optimizer = torch.optim.AdamW(gp.parameters(), lr=lr)

    # None inputs not allowed
    x_train = tuple([x for x in train_inputs if x is not None])
    y_train = train_targets

    for _ in trange(n_steps, disable=not progress_bar):
        optimizer.zero_grad()
        output = gp(*x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
    return gp, likelihood