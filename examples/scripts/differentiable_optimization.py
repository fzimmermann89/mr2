# %% [markdown]
# # Learning through a nonlinear qMRI fit
#
# We train a small CNN through an L-BFGS fit of a mono-exponential signal model.
# A residual CNN corrects the echo images before the fit. We compute the training
# loss on the fitted $M_0$ and $T_2$ maps, learning the CNN weights and the fit's
# regularization strength together. The use of a nonlinear fit as a trainable layer
# follows the example in
# [ImplicitOptimLayer](https://github.com/fzimmermann89/ImplicitOptimLayer/blob/master/nonlinearlayer.ipynb).
#
# We use random 32 x 32 ellipse phantoms and 40 training steps to keep the example
# small enough for a short CPU run. We simulate noisy echo images and learn an
# image-space correction; we do not simulate a k-space acquisition.

# %%
from functools import partial
from time import perf_counter

import matplotlib.pyplot as plt
import torch
from mr2.algorithms.optimizers import lbfgs
from mr2.data import SpatialDimension
from mr2.operators import ConstraintsOp, FunctionalType, OptimizerOp
from mr2.operators.functionals import L2NormSquared
from mr2.operators.models import MonoExponentialDecay
from mr2.phantoms import EllipseParameters, EllipsePhantom

# Small tensors run faster with one CPU thread. Restore the previous setting below.
previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
started = perf_counter()
dtype = torch.float64
image_size = SpatialDimension(z=1, y=32, x=32)
echo_times = torch.linspace(0.01, 0.12, 6, dtype=dtype)
signal_model = MonoExponentialDecay(echo_times)

# %% [markdown]
# ## Generate training and validation phantoms
#
# Each phantom has three randomly placed ellipses with different amplitudes and
# relaxation times. The two parameter maps share the ellipse boundaries. We add
# Gaussian noise with standard deviation 0.025 to the simulated signals
# $q_t(M_0,T_2)=M_0\exp(-t/T_2)$.
# We use seconds for echo times and $T_2$, and arbitrary units for signal amplitude.
# Training and validation use separate random seeds.


# %%
def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate noisy echoes and parameter maps for random ellipse phantoms."""
    rng = torch.Generator().manual_seed(seed)
    m0_maps, t2_maps = [], []
    for _ in range(count):
        geometry = torch.rand((3, 4), generator=rng)
        m0_ellipses, t2_ellipses = [], []
        for center_x, center_y, radius_x, radius_y in geometry:
            shape = {
                'center_x': 0.5 * (center_x.item() - 0.5),
                'center_y': 0.5 * (center_y.item() - 0.5),
                'radius_x': 0.08 + 0.13 * radius_x.item(),
                'radius_y': 0.08 + 0.13 * radius_y.item(),
            }
            m0_ellipses.append(EllipseParameters(**shape, intensity=0.08 + 0.12 * torch.rand((), generator=rng).item()))
            t2_ellipses.append(
                EllipseParameters(**shape, intensity=0.01 + 0.015 * torch.rand((), generator=rng).item())
            )
        m0_maps.append(0.7 + EllipsePhantom(m0_ellipses).image_space(image_size).real.squeeze())
        t2_maps.append(0.055 + EllipsePhantom(t2_ellipses).image_space(image_size).real.squeeze())
    m0, t2 = torch.stack(m0_maps).to(dtype), torch.stack(t2_maps).to(dtype)
    (signal,) = signal_model(m0, t2)  # (echo, phantom, y, x)
    noisy = signal + 0.025 * torch.randn(signal.shape, generator=rng, dtype=dtype)
    return noisy, m0, t2


train_data, train_m0, train_t2 = make_data(8, seed=7)
validation_data, validation_m0, validation_t2 = make_data(4, seed=19)

# %% [markdown]
# ## Set up the nonlinear fitting layer
#
# `~mr2.operators.ConstraintsOp` maps dimensionless variables $u,v$ to physical
# parameters using sigmoid transforms: $M_0\in(0.5,1.5)$ and $T_2\in(0.03,0.18)$ s.
# For CNN-corrected echo images $\widetilde y$, the inner fit minimizes
#
# $$F(u,v;\widetilde y,\alpha)=\tfrac12\|q(C(u,v))-\widetilde y\|_2^2
#       +\tfrac\alpha2\big(\|u\|_2^2+\|v\|_2^2\big).$$
#
# The quadratic term pulls the fit towards the midpoints of the parameter ranges.
#
# We use `~mr2.operators.OptimizerOp` with an objective factory and an initializer.
# Both receive the corrected observations and weight. ``@`` composes the squared norm
# with the nonlinear model. ``|`` adds functionals acting on separate inputs.
# The objective returns a one-element tuple containing its scalar value;
# the fit returns a tuple with the two fitted variables.

# %%
constraints = ConstraintsOp(bounds=((0.5, 1.5), (0.03, 0.18))).double()
constrained_model = signal_model @ constraints


def objective_factory(observations: torch.Tensor, alpha: torch.Tensor) -> FunctionalType:
    """Build the nonlinear least-squares objective for the corrected echoes."""
    data_term = 0.5 * L2NormSquared(observations) @ constrained_model
    regularizer = alpha * (0.5 * L2NormSquared() | 0.5 * L2NormSquared())
    return data_term + regularizer


def initializer(observations: torch.Tensor, _alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Start at the midpoints of the physical parameter ranges."""
    return torch.zeros_like(observations[0]), torch.zeros_like(observations[0])


fit = OptimizerOp(
    factory=objective_factory,
    initializer=initializer,
    optimize=partial(lbfgs, max_iterations=40, max_evaluations=60, tolerance_grad=1e-7, tolerance_change=1e-10),
)

# %% [markdown]
# The L-BFGS function detaches its initial values. We pass the corrected echoes and
# learnable weight as explicit fit inputs. The CNN receives gradients through the
# data term in the objective. Using its output only as an initializer would not
# provide this gradient.
#
# ## Check the implicit derivative on one voxel
#
# At a stationary solution $w^*=(u^*,v^*)$, with objective parameters
# $p=(\widetilde y,\alpha)$, implicit differentiation solves
#
# $$H^T z=\nabla_w L,\qquad
#   \nabla_p L=-\left(\frac{\partial\nabla_w F}{\partial p}\right)^Tz,
#   \quad H=\nabla_w^2 F(w^*,p).$$
#
# `OptimizerOp` uses Hessian-vector products and CG for this backward solve.
# Its current backward uses at most 50 iterations and an absolute residual tolerance
# of $10^{-6}$. The L-BFGS options above control only the forward fit.
# Both solves must converge, and the Hessian must be positive definite for CG.
# The nonlinear objective need not be convex everywhere; this derivative follows a
# local stationary solution and does not describe jumps between different minima.
#
# Unrolling would differentiate individual optimizer updates and retain their graph.
# Here we differentiate the stationarity equation. Calling MRtwo's L-BFGS function
# directly returns detached results, so it does not provide an unrolled derivative.
# We check stationarity and Hessian eigenvalues on one voxel, then compare the
# derivatives with respect to the weight and one echo with centered finite
# differences using tighter forward settings.


# %%
def parameter_error(
    m0: torch.Tensor, t2: torch.Tensor, target_m0: torch.Tensor, target_t2: torch.Tensor
) -> torch.Tensor:
    """Normalize parameter errors by the widths of their physical ranges."""
    return ((m0 - target_m0).square() + ((t2 - target_t2) / 0.15).square()).mean()


check_fit = OptimizerOp(
    objective_factory,
    initializer,
    optimize=partial(lbfgs, max_iterations=100, max_evaluations=150, tolerance_grad=1e-11, tolerance_change=1e-15),
)
center = image_size.x // 2
voxel = train_data[:, 0, center, center].unsqueeze(1).clone().requires_grad_(True)
alpha_check = torch.tensor(0.05, dtype=dtype, requires_grad=True)
check_u, check_v = check_fit(voxel, alpha_check)
check_m0, check_t2 = constraints(check_u, check_v)
target_m0, target_t2 = train_m0[0, center, center], train_t2[0, center, center]
check_loss = parameter_error(check_m0, check_t2, target_m0, target_t2)
implicit_gradient, signal_gradient = torch.autograd.grad(check_loss, (alpha_check, voxel))
point = torch.stack((check_u, check_v)).detach().requires_grad_(True)
objective = objective_factory(voxel.detach(), alpha_check)
(value,) = objective(*point.unbind())
(stationarity,) = torch.autograd.grad(value, point)
hessian = torch.autograd.functional.hessian(lambda w: objective(*w.unbind())[0], point).reshape(2, 2)
assert stationarity.abs().max() < 1e-6
assert torch.linalg.eigvalsh(hessian).min() > 0
step = 1e-3
plus_m0, plus_t2 = constraints(*check_fit(voxel.detach(), alpha_check.detach() + step))
minus_m0, minus_t2 = constraints(*check_fit(voxel.detach(), alpha_check.detach() - step))
finite_difference = (
    parameter_error(plus_m0, plus_t2, target_m0, target_t2) - parameter_error(minus_m0, minus_t2, target_m0, target_t2)
) / (2 * step)
torch.testing.assert_close(implicit_gradient, finite_difference, rtol=2e-3, atol=1e-6)
echo_step = torch.zeros_like(voxel)
echo_step[2] = 1e-4
plus_m0, plus_t2 = constraints(*check_fit(voxel.detach() + echo_step, alpha_check.detach()))
minus_m0, minus_t2 = constraints(*check_fit(voxel.detach() - echo_step, alpha_check.detach()))
echo_difference = (
    parameter_error(plus_m0, plus_t2, target_m0, target_t2) - parameter_error(minus_m0, minus_t2, target_m0, target_t2)
) / (2e-4)
torch.testing.assert_close(signal_gradient[2, 0], echo_difference, rtol=2e-3, atol=1e-6)
{
    'weight gradient': implicit_gradient.item(),
    'weight finite difference': finite_difference.item(),
    'echo gradient': signal_gradient[2, 0].item(),
    'echo finite difference': echo_difference.item(),
}

# %% [markdown]
# ## Train a CNN through the fitted parameter maps
#
# The CNN takes six echo images as channels and predicts an additive correction.
# We initialize the last convolution to zero, so the initial fit receives the noisy
# images unchanged. We train with Adam on batches of two phantoms. The outer loss
# uses fitted parameters, normalized by their range widths:
#
# $$L=\operatorname{mean}\left[(M_0^*-M_{0,\mathrm{true}})^2
#             +\left(\frac{T_2^*-T_{2,\mathrm{true}}}{0.15\,\mathrm{s}}\right)^2\right].$$
#
# We optimize $\alpha=\exp(\theta)$ to keep the regularization weight positive.
# The synthetic targets enter only the outer loss. With parameter-only supervision,
# the CNN can compensate for the fit's regularization; its corrected echoes need
# not equal the noise-free signals. We evaluate the fixed validation set before and
# after training. These small phantoms do not establish performance on measured MRI.

# %%
with torch.random.fork_rng():
    torch.manual_seed(7)
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(6, 8, 3, padding=1), torch.nn.Tanh(), torch.nn.Conv2d(8, 6, 3, padding=1)
    ).double()
with torch.no_grad():
    for parameter in cnn[-1].parameters():
        parameter.zero_()
log_alpha = torch.nn.Parameter(torch.tensor(-3.0, dtype=dtype))
outer_optimizer = torch.optim.Adam([*cnn.parameters(), log_alpha], lr=0.003)

# CNN layout: (phantom, echo, y, x). Signal-model layout: (echo, phantom, y, x).
initial_u, initial_v = fit(validation_data, log_alpha.exp().detach())
initial_m0, initial_t2 = (x.detach() for x in constraints(initial_u, initial_v))
initial_error = parameter_error(initial_m0, initial_t2, validation_m0, validation_t2)

training_started = perf_counter()
loss_history = []
for iteration in range(40):
    indices = torch.arange(2) + 2 * (iteration % 4)
    observations = train_data[:, indices]
    outer_optimizer.zero_grad()
    corrected = observations + cnn(observations.movedim(0, 1)).movedim(1, 0)
    u, v = fit(corrected, log_alpha.exp())
    m0, t2 = constraints(u, v)
    loss = parameter_error(m0, t2, train_m0[indices], train_t2[indices])
    loss_history.append(loss.item())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cnn.parameters())
    assert log_alpha.grad is not None and torch.isfinite(log_alpha.grad)
    outer_optimizer.step()
training_seconds = perf_counter() - training_started

# L-BFGS uses autograd internally, so keep it enabled when evaluating the fit.
corrected_validation = validation_data + cnn(validation_data.movedim(0, 1)).movedim(1, 0).detach()
learned_u, learned_v = fit(corrected_validation, log_alpha.exp().detach())
learned_m0, learned_t2 = (x.detach() for x in constraints(learned_u, learned_v))
learned_error = parameter_error(learned_m0, learned_t2, validation_m0, validation_t2)
assert learned_error < initial_error
# Check stationarity of the complete validation fit, not only the one-voxel example.
validation_point = tuple(x.detach().requires_grad_(True) for x in (learned_u, learned_v))
validation_objective = objective_factory(corrected_validation, log_alpha.exp().detach())
validation_gradient = torch.autograd.grad(validation_objective(*validation_point)[0], validation_point)
max_validation_residual = max(g.abs().max().item() for g in validation_gradient)
assert max_validation_residual < 1e-4
{
    'initial validation error': initial_error.item(),
    'trained validation error': learned_error.item(),
    'learned alpha': log_alpha.exp().item(),
    'training seconds': training_seconds,
    'max validation stationarity residual': max_validation_residual,
}

# %% [markdown]
# ## Compare validation maps
#
# We show one phantom from the validation set, using the same color scale for all
# $T_2$ maps. The initial fit uses the noisy echoes. After training, the fit uses
# the CNN-corrected echoes and learned weight. The loss curve shows the successive
# training batches, so adjacent points may correspond to different phantoms.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show plotting code"}
fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
for axis, image, title in zip(
    axes[0],
    (validation_t2[0], initial_t2[0], learned_t2[0]),
    ('Target $T_2$', 'Initial fit', 'Trained CNN + fit'),
    strict=True,
):
    im = axis.imshow(1000 * image, vmin=45, vmax=130, cmap='viridis')
    axis.set_title(title)
    axis.set_axis_off()
fig.colorbar(im, ax=list(axes[0]), label='$T_2$ (ms)', shrink=0.8)
axes[1, 0].imshow(validation_m0[0], cmap='gray', vmin=0.7, vmax=1.3)
axes[1, 0].set_title('Target $M_0$ (a.u.)')
axes[1, 0].set_axis_off()
axes[1, 1].semilogy(loss_history)
axes[1, 1].set(xlabel='Outer step', ylabel='Normalized parameter error', title='Training loss')
error_image = axes[1, 2].imshow(1000 * (learned_t2[0] - validation_t2[0]), cmap='RdBu_r', vmin=-25, vmax=25)
axes[1, 2].set_title('Trained $T_2$ error')
axes[1, 2].set_axis_off()
fig.colorbar(error_image, ax=axes[1, 2], label='Error (ms)', shrink=0.8)
plt.show()
total_seconds = perf_counter() - started
torch.set_num_threads(previous_threads)
{'total compute and plotting seconds': round(total_seconds, 2)}
