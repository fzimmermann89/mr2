# %% [markdown]
# # Training through qMRI parameter estimation
#
# Here we train a CNN to correct noisy echo images using a loss on the fitted
# $M_0$ and $T_2$ maps. We compare two parameter estimation methods:
# `~mr2.operators.OptimizerOp` with L-BFGS and
# `~mr2.operators.DifferentiableDictionaryMatchOp`.
# Both use the same mono-exponential signal model and CNN architecture.
#
# The training chain is
#
# $$y \;\longrightarrow\; \widetilde y=y+f_\theta(y)
#   \;\longrightarrow\; (M_0^*,T_2^*) \;\longrightarrow\; L.$$
#
# The CNN uses neighboring pixels to correct the echoes. The parameter estimator
# then fits each voxel's decay curve. We supervise the fitted parameter maps, so a
# training step must propagate gradients through the estimator back to the CNN.
# The signal model and the dictionary remain fixed during training.
#
# We train on random ellipse phantoms and compare the fitted maps on separate
# validation phantoms at the end. Training uses a fixed number of epochs; runtime
# depends on the CPU and the fitting method.

# %%
from collections.abc import Callable
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import mr2
import torch

dtype = torch.float64
image_size = mr2.data.SpatialDimension(z=1, y=64, x=64)
echo_times = torch.linspace(0.01, 0.15, 8, dtype=dtype)
signal_model = mr2.operators.models.MonoExponentialDecay(echo_times)
batch_size = 1

# %% [markdown]
# ## Generate phantoms
#
# We vary the number, position and size of the ellipses. Each ellipse replaces the
# underlying tissue values, so overlapping ellipses can form both bright and dark
# regions. We draw $M_0$ and $T_2$ independently, including the background values.
# Each map contains between three and twelve ellipses, with $M_0$ between 0.55 and
# 1.4 arbitrary units and $T_2$ between 35 and 180 ms.
#
# We simulate eight echoes using
# $q_t(M_0,T_2)=M_0\exp(-t/T_2)$. $M_0$ controls the amplitude at zero echo time;
# $T_2$ controls how quickly it decays. A longer $T_2$ gives a slower decay.
# Echo times and $T_2$ are in seconds in the code.
# Training uses 32 phantoms with new Gaussian noise at each update. Validation
# uses 8 other phantoms with fixed noise. The noise standard deviation
# varies between 0.015 and 0.045 per phantom. All random generators have fixed seeds.
# We work with echo images here, without simulating a k-space acquisition.


# %%
def make_phantoms(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate clean echoes and parameter maps with overlapping random ellipses."""
    rng = torch.Generator().manual_seed(seed)
    m0_maps, t2_maps = [], []
    for _ in range(count):
        m0 = torch.full((image_size.y, image_size.x), 0.55 + 0.85 * torch.rand((), generator=rng).item())
        t2 = torch.full_like(m0, 0.035 + 0.145 * torch.rand((), generator=rng).item())
        ellipse_count = int(torch.randint(3, 13, (), generator=rng))
        for center_x, center_y, radius_x, radius_y in torch.rand((ellipse_count, 4), generator=rng):
            ellipse = mr2.phantoms.EllipseParameters(
                center_x=0.8 * (center_x.item() - 0.5),
                center_y=0.8 * (center_y.item() - 0.5),
                radius_x=0.025 + 0.235 * radius_x.item(),
                radius_y=0.025 + 0.235 * radius_y.item(),
                intensity=1,
            )
            # Replace both parameter values inside this ellipse, including overlaps.
            mask = mr2.phantoms.EllipsePhantom([ellipse]).image_space(image_size).real.squeeze() > 0
            m0[mask] = 0.55 + 0.85 * torch.rand((), generator=rng).item()
            t2[mask] = 0.035 + 0.145 * torch.rand((), generator=rng).item()
        m0_maps.append(m0)
        t2_maps.append(t2)
    m0, t2 = torch.stack(m0_maps).to(dtype), torch.stack(t2_maps).to(dtype)
    # One model evaluation produces every echo for every voxel in the batch.
    (echoes,) = signal_model(m0, t2)  # (echo, phantom, y, x)
    return echoes, m0, t2


def add_noise(echoes: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Add Gaussian noise with a separate standard deviation for each phantom."""
    sigma = 0.015 + 0.03 * torch.rand((1, echoes.shape[1], 1, 1), generator=rng, dtype=dtype)
    return echoes + sigma * torch.randn(echoes.shape, generator=rng, dtype=dtype)


train_echoes, train_m0, train_t2 = make_phantoms(32, seed=7)
validation_echoes, validation_m0, validation_t2 = make_phantoms(8, seed=19)
validation_data = add_noise(validation_echoes, torch.Generator().manual_seed(20))

# %% [markdown]
# Here are four training phantoms, with $M_0$ in the first row and $T_2$ in the
# second row. We use the same color scales for all four phantoms.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show phantom plotting code"}
fig, axes = plt.subplots(2, 4, figsize=(10, 5), layout='constrained')
for index in range(4):
    m0_image = axes[0, index].imshow(train_m0[index], vmin=0.55, vmax=1.4, cmap='gray')
    t2_image = axes[1, index].imshow(1000 * train_t2[index], vmin=35, vmax=180, cmap='viridis')
    axes[0, index].set_title(f'Phantom {index + 1}')
    axes[0, index].set_axis_off()
    axes[1, index].set_axis_off()
fig.colorbar(m0_image, ax=list(axes[0]), label='$M_0$ (a.u.)', shrink=0.8)
fig.colorbar(t2_image, ax=list(axes[1]), label='$T_2$ (ms)', shrink=0.8)
plt.show()

# %% [markdown]
# ## Fit the parameters with L-BFGS
#
# We first use `~mr2.operators.OptimizerOp` to minimize a nonlinear least-squares
# objective. L-BFGS repeatedly evaluates this objective and its gradient. It uses a
# short history of parameter and gradient differences to approximate inverse
# curvature, and a line search chooses how far to move along each search direction.
# This fits the parameters without forming a full Hessian in the forward pass.
#
# The inner objective measures agreement with the corrected echo images. It has
# no access to the target parameter maps. The outer training loss, defined below,
# measures parameter error and updates the CNN.
#
# `~mr2.operators.ConstraintsOp` maps dimensionless variables $u,v$ to
# $M_0\in(0.3,1.8)$ and $T_2\in(0.015,0.23)$ s using sigmoid transforms.
# For corrected echo images $\widetilde y$, the fit minimizes
#
# $$F(u,v;\widetilde y)=\tfrac12\operatorname{MSE}(q(C(u,v)),\widetilde y)
#     +\tfrac\alpha2\big(\operatorname{MSE}(u,0)+\operatorname{MSE}(v,0)\big).$$
#
# We use `~mr2.operators.functionals.MSE` and fix $\alpha=1.25\times10^{-4}$.
# The quadratic term weakly regularizes the fit towards the parameter midpoints.
# We multiply the objective by the number of voxels to keep each voxel's gradient
# scale independent of batch size. This constant does not change the minimizer;
# absolute function-change stopping criteria can still depend on the total size.
#
# The factory builds the objective. The initializer fits a straight line to the log
# of the first four echoes and converts it to starting values for $M_0$ and $T_2$.
# For a noise-free exponential, $\log q_t=\log M_0-t/T_2$, so the line's slope
# gives $-1/T_2$ and its intercept gives $\log M_0$. We use the earlier echoes
# because their signal is larger, and clamp the estimates to stay inside the bounds.
# This is a starting guess; L-BFGS fits the original signal values, not their logs.
#
# Both factory and initializer receive the fit inputs. The factory returns a
# callable objective; the initializer returns the two tensors to optimize.
# ``@`` composes operators, and ``|`` adds scalar functionals acting on separate
# inputs. We call operators as ``op(x)`` and unpack their tuple outputs.

# %%
constraints = mr2.operators.ConstraintsOp(bounds=((0.3, 1.8), (0.015, 0.23))).double()
constrained_model = signal_model @ constraints
alpha = torch.tensor(1.25e-4, dtype=dtype)


def objective_factory(observations: torch.Tensor, weight: torch.Tensor) -> mr2.operators.FunctionalType:
    """Build the nonlinear fitting objective for the corrected echoes."""
    # Apply C, evaluate the decay model, then compare predicted and corrected echoes.
    data_term = mr2.operators.functionals.MSE(observations) @ constrained_model
    # Each MSE acts on one unconstrained parameter map; | adds the two scalars.
    regularizer = weight * (mr2.operators.functionals.MSE() | mr2.operators.functionals.MSE())
    return (0.5 * observations[0].numel()) * (data_term + regularizer)


def initializer(observations: torch.Tensor, _weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize from a log-linear fit to the first four echoes."""
    times = echo_times[:4].reshape((-1,) + (1,) * (observations.ndim - 1))
    centered_times = times - times.mean()
    # This supplies a constant warm start; gradients come through the objective.
    log_signal = observations[:4].detach().clamp_min(1e-3).log()
    slope = (centered_times * log_signal).sum(0) / centered_times.square().sum()
    t2 = (-slope).clamp_min(1 / 0.21).reciprocal().clamp(0.025, 0.21)
    m0 = (log_signal.mean(0) + times.mean() / t2).exp().clamp(0.35, 1.75)
    u, v = constraints.inverse(m0, t2)
    return u, v


fit = mr2.operators.OptimizerOp(
    objective_factory,
    initializer,
    optimize=partial(
        mr2.algorithms.optimizers.lbfgs,
        max_iterations=60,
        max_evaluations=90,
        tolerance_grad=1e-7,
        tolerance_change=1e-11,
    ),
)


def lbfgs_parameters(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate physical parameters with the differentiable L-BFGS layer."""
    # The optimizer returns unconstrained variables; convert them to physical units.
    u, v = fit(observations, alpha)
    m0, t2 = constraints(u, v)
    return m0, t2


# %% [markdown]
# The forward pass runs L-BFGS. In the backward pass, `OptimizerOp` differentiates
# the stationarity equation at the fitted solution $w^*=(u^*,v^*)$:
#
# $$H^Tz=\nabla_w L,\qquad
# \nabla_{\widetilde y}L=-\left(
# \frac{\partial\nabla_w F}{\partial\widetilde y}\right)^Tz,
# \qquad H=\nabla_w^2F(w^*,\widetilde y).$$
#
# $\nabla_w L$ tells us how the outer loss changes with the fitted variables.
# Solving for $z$ accounts for how sensitive the optimum is to changes in the
# objective. The mixed derivative then maps that sensitivity to the input echoes.
# Autograd computes these derivatives from the same objective factory.
#
# `OptimizerOp` uses CG and Hessian-vector products for this backward solve. The current
# implementation uses at most 50 CG iterations with absolute residual tolerance
# $10^{-6}$, independently of the forward L-BFGS settings. Both solves must converge,
# and the Hessian must be positive definite for CG. The derivative follows a local
# stationary solution; it does not describe jumps between different minima.
#
# Unrolling would differentiate the optimizer updates and store their graph.
# Here we differentiate the stationarity equation. MRtwo's L-BFGS function returns
# detached results, so calling it directly does not give an unrolled derivative.
# Initial values are also detached. The CNN receives gradients because its corrected
# echoes enter the objective, rather than only providing an initialization.

# %% [markdown]
# ## Estimate the parameters with dictionary matching
#
# `~mr2.operators.DifferentiableDictionaryMatchOp` first finds the dictionary signal
# with the largest normalized correlation. It estimates $M_0$ as a signal scaling
# factor, then takes one Gauss-Newton step using precomputed signal derivatives.
# We store 96 $T_2$ values covering the phantom range with some margin.
# ``index_of_scaling_parameter=0`` identifies $M_0$ as the scaling parameter.
# We therefore tabulate the signal shape at unit amplitude rather than building a
# two-dimensional dictionary over both $M_0$ and $T_2$.
#
# At the selected $T_{2,k}$, the operator first estimates
# $M_{0,k}=q(1,T_{2,k})^T\widetilde y/\|q(1,T_{2,k})\|_2^2$.
# With $p_k=(M_{0,k},T_{2,k})$ and signal Jacobian $J_k$, it then solves the local
# least-squares problem
#
# $$(J_k^T J_k)\Delta p=J_k^T(\widetilde y-q(p_k)),
#       \qquad p_{\mathrm{out}}=p_k+\Delta p.$$
#
# This refinement gives continuous parameter values rather than a grid index.
# It does not enforce bounds, so a large update can also leave the dictionary range.
# It uses one linear approximation around the selected signal, whereas L-BFGS
# repeatedly reevaluates the nonlinear objective while updating the parameters.
#
# The dictionary lookup is discrete and has no derivative. Autograd differentiates
# the refinement with the selected entry held fixed. This gives a local derivative
# with respect to the corrected echoes, not a smooth derivative across changes in
# the selected entry. One refinement step also need not reach a nonlinear minimum.

# %%
dictionary_match = mr2.operators.DifferentiableDictionaryMatchOp[torch.Tensor, torch.Tensor](
    signal_model, index_of_scaling_parameter=0, batch_size=4096
)
t2_grid = torch.linspace(0.025, 0.22, 96, dtype=dtype)
# append computes dictionary signals and their derivatives once, before training.
dictionary_match.append(torch.ones_like(t2_grid), t2_grid)


def dictionary_parameters(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match and refine the corrected echoes without compilation startup time."""
    # The search kernel is normally compiled. Eager execution suits this short CPU run.
    with torch.compiler.set_stance('force_eager'):
        m0, t2 = dictionary_match(observations)
    return m0, t2


# %% [markdown]
# ## Define the parameter loss
#
# We compare the fitted maps with their targets using MSE. Dividing by the widths
# of the fitting ranges makes both errors dimensionless, so the units of $T_2$
# do not determine its weight relative to $M_0$.


# %%
def parameter_error(
    m0: torch.Tensor, t2: torch.Tensor, target_m0: torch.Tensor, target_t2: torch.Tensor
) -> torch.Tensor:
    """Normalize squared errors by the widths of the fitting ranges."""
    (m0_error,) = mr2.operators.functionals.MSE(target_m0, weight=1 / 1.5)(m0)
    (t2_error,) = mr2.operators.functionals.MSE(target_t2, weight=1 / 0.215)(t2)
    return m0_error + t2_error


# %% [markdown]
# ## Define the CNN
#
# The CNN treats the eight echoes as channels and predicts an additive correction.
# The final convolution starts at zero, so the initial correction is zero.
# The signal model expects the echo axis first; we move it to the channel position
# for the CNN and move it back before fitting.
#
# We train separate copies of this network for the two estimators. Only the CNN
# weights change. Parameter-only supervision can compensate for fitting bias,
# so the corrected echoes need not equal the noise-free signals.

# %%
Estimator = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
with torch.random.fork_rng():
    torch.manual_seed(42)
    initial_cnn = torch.nn.Sequential(
        torch.nn.Conv2d(8, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 8, 3, padding=1),
    ).double()
with torch.no_grad():
    for parameter in initial_cnn[-1].parameters():
        parameter.zero_()


# %% [markdown]
# ## Train through a parameter estimator
#
# Each batch passes through the CNN and then the parameter estimator. The loss
# compares the fitted maps with the training targets. Calling ``backward`` sends
# gradients through the estimator to the CNN; Adam then updates the CNN weights.
# We shuffle the training phantoms and draw fresh noise each epoch.


# %%
def train(estimate: Estimator, epochs: int) -> tuple[torch.nn.Module, list[float]]:
    """Train a copy of the CNN and record mean loss for each epoch."""
    cnn = deepcopy(initial_cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    rng = torch.Generator().manual_seed(100)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for indices in torch.randperm(len(train_m0), generator=rng).split(batch_size):
            observations = add_noise(train_echoes[:, indices], rng)
            optimizer.zero_grad()
            # Correct the echoes, fit each voxel, then supervise the parameter maps.
            corrected = observations + cnn(observations.movedim(0, 1)).movedim(1, 0)
            m0, t2 = estimate(corrected)
            loss = parameter_error(m0, t2, train_m0[indices], train_t2[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)
        losses.append(epoch_loss / len(train_m0))
        print(f'Epoch {epoch + 1}: loss = {losses[-1]:.5f}')
    return cnn, losses


# %% [markdown]
# ## Train with L-BFGS
#
# We run two epochs through the nonlinear fitting layer. Its backward pass uses
# implicit differentiation. The objective, bounds and regularization stay fixed.

# %%
lbfgs_cnn, lbfgs_losses = train(lbfgs_parameters, epochs=2)

# %% [markdown]
# ## Train with dictionary matching
#
# We train a fresh copy of the same CNN through the dictionary refinement.
# Matching is cheaper per batch, so we run 20 epochs. The dictionary stays fixed.
# These different update counts illustrate both approaches rather than a controlled
# comparison of their training speed.

# %%
dictionary_cnn, dictionary_losses = train(dictionary_parameters, epochs=20)

# %% [markdown]
# ## Plot the training loss
#
# Each point is the mean parameter loss over one training epoch.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show loss plotting code"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout='constrained')
for axis, losses, title in zip(
    axes, (lbfgs_losses, dictionary_losses), ('CNN + L-BFGS', 'CNN + dictionary'), strict=True
):
    axis.semilogy(range(1, len(losses) + 1), losses, '.-')
    axis.set(xlabel='Epoch', ylabel='Normalized parameter error', title=title)
plt.show()

# %% [markdown]
# ## Compare validation examples
#
# After training, we apply both CNNs to the same noisy validation images.
# We use the final weights directly. The validation targets enter only the metrics
# and plots below. `~mr2.operators.functionals.L1Norm` with averaging gives mean
# absolute error; `~mr2.operators.functionals.MSE` gives the normalized squared error.
#
# The two estimators solve different fitting problems: L-BFGS includes a quadratic
# regularizer, whereas dictionary matching uses a discrete search and one refinement
# step. These phantoms therefore illustrate training through each estimator, rather
# than establishing their relative performance on measured MRI data.

# %%
validation_predictions = {}
validation_metrics = {}
for name, cnn, estimate in (
    ('L-BFGS', lbfgs_cnn, lbfgs_parameters),
    ('Dictionary', dictionary_cnn, dictionary_parameters),
):
    # Disable the CNN graph; L-BFGS still needs autograd inside its forward fit.
    with torch.no_grad():
        corrected = validation_data + cnn(validation_data.movedim(0, 1)).movedim(1, 0)
    m0, t2 = (value.detach() for value in estimate(corrected))
    validation_predictions[name] = (m0, t2)
    validation_metrics[name] = {
        'M0 MAE (a.u.)': mr2.operators.functionals.L1Norm(validation_m0, divide_by_n=True)(m0)[0].item(),
        'T2 MAE (ms)': 1000 * mr2.operators.functionals.L1Norm(validation_t2, divide_by_n=True)(t2)[0].item(),
        'normalized parameter error': parameter_error(m0, t2, validation_m0, validation_t2).item(),
    }

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show metric table code"}
fig, axis = plt.subplots(figsize=(10, 1.4), layout='constrained')
axis.set_axis_off()
metric_rows = [
    [
        name,
        f'{metrics["M0 MAE (a.u.)"]:.4f}',
        f'{metrics["T2 MAE (ms)"]:.2f}',
        f'{metrics["normalized parameter error"]:.5f}',
    ]
    for name, metrics in validation_metrics.items()
]
table = axis.table(
    cellText=metric_rows,
    colLabels=('Estimator', '$M_0$ MAE (a.u.)', '$T_2$ MAE (ms)', 'Normalized error'),
    colWidths=(0.32, 0.23, 0.23, 0.22),
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
plt.show()

# %% [markdown]
# We show three validation phantoms with different tissue values. All $T_2$ maps use the
# same color scale. The last two columns show the error of each trained estimator.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show validation plotting code"}
fig, axes = plt.subplots(3, 5, figsize=(13, 8), layout='constrained')
for row, index in enumerate((0, 3, 6)):
    maps = (
        validation_t2[index],
        validation_predictions['L-BFGS'][1][index],
        validation_predictions['Dictionary'][1][index],
        validation_predictions['L-BFGS'][1][index] - validation_t2[index],
        validation_predictions['Dictionary'][1][index] - validation_t2[index],
    )
    for column, (axis, image) in enumerate(zip(axes[row], maps, strict=True)):
        if column < 3:
            t2_image = axis.imshow(1000 * image, vmin=35, vmax=180, cmap='viridis')
        else:
            error_image = axis.imshow(1000 * image, vmin=-30, vmax=30, cmap='RdBu_r')
        axis.set_axis_off()
for axis, title in zip(
    axes[0], ('Target $T_2$', 'CNN + L-BFGS', 'CNN + dictionary', 'L-BFGS error', 'Dictionary error'), strict=True
):
    axis.set_title(title)
fig.colorbar(t2_image, ax=list(axes[:, :3].flatten()), label='$T_2$ (ms)', shrink=0.6)
fig.colorbar(error_image, ax=list(axes[:, 3:].flatten()), label='Error (ms)', shrink=0.6)
plt.show()
