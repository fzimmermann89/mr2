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
# We generate random ellipse phantoms and train each CNN for 60 seconds on the CPU.
# Validation selects a checkpoint; a separate test set measures the final errors.
# The notebook takes a few minutes, including validation and plotting.

# %%
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from time import perf_counter

import matplotlib.pyplot as plt
import mr2
import torch

# One CPU thread avoids parallelization overhead for these small images.
previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
dtype = torch.float64
image_size = mr2.data.SpatialDimension(z=1, y=64, x=64)
echo_times = torch.linspace(0.01, 0.15, 8, dtype=dtype)
signal_model = mr2.operators.models.MonoExponentialDecay(echo_times)
training_seconds = 60.0
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
# Training uses 128 phantoms with new Gaussian noise at each update. Validation
# and test each use 16 other phantoms with fixed noise. The noise standard deviation
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


train_echoes, train_m0, train_t2 = make_phantoms(128, seed=7)
validation_echoes, validation_m0, validation_t2 = make_phantoms(16, seed=19)
test_echoes, test_m0, test_t2 = make_phantoms(16, seed=47)
validation_data = add_noise(validation_echoes, torch.Generator().manual_seed(20))
test_data = add_noise(test_echoes, torch.Generator().manual_seed(48))

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
# ## Check the derivatives
#
# On one voxel, we compare the L-BFGS layer's implicit derivative with a centered
# finite difference. We also check forward stationarity and Hessian eigenvalues.
# For dictionary matching, we check recovery of a noise-free dictionary signal and
# its input derivative. We choose a support point away from a lookup boundary so
# the numerical perturbations keep the same dictionary entry.


# %%
def parameter_error(
    m0: torch.Tensor, t2: torch.Tensor, target_m0: torch.Tensor, target_t2: torch.Tensor
) -> torch.Tensor:
    """Normalize squared errors by the widths of the fitting ranges."""
    (m0_error,) = mr2.operators.functionals.MSE(target_m0, weight=1 / 1.5)(m0)
    (t2_error,) = mr2.operators.functionals.MSE(target_t2, weight=1 / 0.215)(t2)
    return m0_error + t2_error


check_fit = mr2.operators.OptimizerOp(
    objective_factory,
    initializer,
    optimize=partial(
        mr2.algorithms.optimizers.lbfgs,
        max_iterations=150,
        max_evaluations=200,
        tolerance_grad=1e-11,
        tolerance_change=1e-15,
    ),
)
# Keep the input differentiable, just as the CNN output will be during training.
voxel = validation_data[:, 0, 32, 32].unsqueeze(1).clone().requires_grad_(True)
u, v = check_fit(voxel, alpha)
m0, t2 = constraints(u, v)
loss = parameter_error(m0, t2, validation_m0[0, 32, 32], validation_t2[0, 32, 32])
(derivative,) = torch.autograd.grad(loss, voxel)
# For two variables, we can form the Hessian to check the local CG assumptions.
point = torch.stack((u, v)).detach().requires_grad_(True)
objective = objective_factory(voxel.detach(), alpha)
(stationarity,) = torch.autograd.grad(objective(*point.unbind())[0], point)
hessian = torch.autograd.functional.hessian(lambda w: objective(*w.unbind())[0], point).reshape(2, 2)
assert stationarity.abs().max() < 1e-6
assert torch.linalg.eigvalsh(hessian).min() > 0
# Perturb only the third echo and refit to approximate dL/dy[2].
perturbation = torch.zeros_like(voxel)
perturbation[2] = 1e-4
plus_m0, plus_t2 = constraints(*check_fit(voxel.detach() + perturbation, alpha))
minus_m0, minus_t2 = constraints(*check_fit(voxel.detach() - perturbation, alpha))
finite_difference = (
    parameter_error(plus_m0, plus_t2, validation_m0[0, 32, 32], validation_t2[0, 32, 32])
    - parameter_error(minus_m0, minus_t2, validation_m0[0, 32, 32], validation_t2[0, 32, 32])
) / (2e-4)
torch.testing.assert_close(derivative[2, 0], finite_difference, rtol=2e-3, atol=1e-6)

reference_m0 = torch.tensor([0.95], dtype=dtype)
reference_t2 = t2_grid[40:41]
(reference_signal,) = signal_model(reference_m0, reference_t2)
matched_m0, matched_t2 = dictionary_parameters(reference_signal)
torch.testing.assert_close(matched_m0, reference_m0, rtol=1e-8, atol=1e-8)
torch.testing.assert_close(matched_t2, reference_t2, rtol=1e-8, atol=1e-8)
# gradcheck compares the refined output Jacobian with numerical derivatives.
assert torch.autograd.gradcheck(dictionary_parameters, (reference_signal.requires_grad_(True),), fast_mode=True)
{'L-BFGS echo gradient': derivative[2, 0].item(), 'finite difference': finite_difference.item()}

# %% [markdown]
# ## Define the CNN and validation loop
#
# The CNN treats the eight echoes as channels. Its convolutions share weights
# across spatial positions and use neighboring pixels to predict an additive echo
# correction. The signal model instead expects the echo axis first, so we move
# the axes before and after each CNN call.
#
# We use the fitted parameters to compute the outer loss
#
# $$L=\operatorname{mean}\left[
# \left(\frac{M_0^*-M_{0,\mathrm{true}}}{1.5\,\mathrm{a.u.}}\right)^2+
# \left(\frac{T_2^*-T_{2,\mathrm{true}}}{0.215\,\mathrm{s}}\right)^2\right].$$
#
# Dividing by the widths of the fitting ranges makes the two squared errors
# dimensionless; otherwise the numerical scales of $M_0$ and $T_2$ would determine
# their relative weight. The training targets enter only this outer loss.
#
# We train separate copies of the same initialized CNN with Adam. The final
# convolution starts at zero, so both CNNs initially pass the noisy echoes through.
# Training shuffles the phantoms each epoch and draws new noise for every batch.
# At each epoch boundary, we evaluate the full validation set and save the weights
# if validation error improves. We also validate at the end of the time budget.
#
# Each method receives 60 seconds of training updates. Validation time is recorded
# separately. The number of updates depends on the CPU and the fitting method, so
# this is a comparison at a fixed training time rather than a fixed update count.
# The random seeds are fixed, but the time limit makes the final update count vary.
# Parameter-only supervision can also let the CNN compensate for fitting bias;
# the corrected echoes need not equal the noise-free signals.

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


def evaluate(
    cnn: torch.nn.Module,
    estimate: Estimator,
    observations: torch.Tensor,
    target_m0: torch.Tensor,
    target_t2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate a complete split in batches, with no gradient through the CNN."""
    cnn.eval()
    m0_batches, t2_batches = [], []
    for start in range(0, observations.shape[1], batch_size):
        batch = observations[:, start : start + batch_size]
        # Disable the CNN graph for validation, then let the fitter use autograd.
        with torch.no_grad():
            corrected = batch + cnn(batch.movedim(0, 1)).movedim(1, 0)
        # L-BFGS needs autograd internally even during evaluation.
        m0, t2 = estimate(corrected)
        m0_batches.append(m0.detach())
        t2_batches.append(t2.detach())
    result_m0, result_t2 = torch.cat(m0_batches), torch.cat(t2_batches)
    assert torch.isfinite(parameter_error(result_m0, result_t2, target_m0, target_t2))
    return result_m0, result_t2


def train(estimate: Estimator) -> tuple[torch.nn.Module, list[tuple[float, float, float]], int, float]:
    """Train for the CPU time budget and return the best validation checkpoint."""
    cnn = deepcopy(initial_cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    sampling_rng = torch.Generator().manual_seed(100)
    noise_rng = torch.Generator().manual_seed(101)
    validation_started = perf_counter()
    m0, t2 = evaluate(cnn, estimate, validation_data, validation_m0, validation_t2)
    best_error = parameter_error(m0, t2, validation_m0, validation_t2).item()
    # Keep a copy so later optimizer updates cannot modify the saved checkpoint.
    best_state = deepcopy(cnn.state_dict())
    validation_seconds = perf_counter() - validation_started
    history = [(0.0, float('nan'), best_error)]
    elapsed = 0.0
    updates = 0
    accumulated_loss = 0.0
    accumulated_updates = 0
    order = torch.randperm(len(train_m0), generator=sampling_rng)
    offset = 0
    while elapsed < training_seconds:
        step_started = perf_counter()
        indices = order[offset : offset + batch_size]
        # Draw fresh observations from the clean training echoes at every update.
        observations = add_noise(train_echoes[:, indices], noise_rng)
        cnn.train()
        optimizer.zero_grad()
        # 1. CNN: echo-first images -> channel-first batch -> corrected echoes.
        corrected = observations + cnn(observations.movedim(0, 1)).movedim(1, 0)
        # 2. Fit each voxel's decay curve; only the CNN uses spatial neighbors.
        m0, t2 = estimate(corrected)
        # 3. Supervise physical parameters, not the corrected echo intensities.
        loss = parameter_error(m0, t2, train_m0[indices], train_t2[indices])
        assert torch.isfinite(loss)
        # This crosses the implicit solve or dictionary refinement to reach the CNN.
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cnn.parameters())
        optimizer.step()  # Update CNN weights; the estimator itself stays fixed.
        accumulated_loss += loss.item()
        accumulated_updates += 1
        updates += 1
        offset += batch_size
        elapsed += perf_counter() - step_started
        if offset == len(order) or elapsed >= training_seconds:
            # Validation never calls backward or changes network weights.
            validation_started = perf_counter()
            m0, t2 = evaluate(cnn, estimate, validation_data, validation_m0, validation_t2)
            validation_error = parameter_error(m0, t2, validation_m0, validation_t2).item()
            history.append((elapsed, accumulated_loss / accumulated_updates, validation_error))
            if validation_error < best_error:
                best_error = validation_error
                best_state = deepcopy(cnn.state_dict())
            validation_seconds += perf_counter() - validation_started
            accumulated_loss = 0.0
            accumulated_updates = 0
            order = torch.randperm(len(train_m0), generator=sampling_rng)
            offset = 0
    # Test evaluation uses this validation-selected checkpoint, not the last update.
    cnn.load_state_dict(best_state)
    return cnn, history, updates, validation_seconds


# %% [markdown]
# ## Train through L-BFGS
#
# We first train through the L-BFGS fitting layer. Only the CNN weights change;
# the signal model, parameter bounds and regularization weight stay fixed.

# %%
lbfgs_cnn, lbfgs_history, lbfgs_updates, lbfgs_validation_seconds = train(lbfgs_parameters)
{
    'training seconds': lbfgs_history[-1][0],
    'updates': lbfgs_updates,
    'validation seconds': lbfgs_validation_seconds,
    'best validation error': min(row[2] for row in lbfgs_history),
}

# %% [markdown]
# ## Train through dictionary matching
#
# We now train another copy of the initial CNN through dictionary matching. We use
# the same learning rate, training phantoms and noise generators. The dictionary
# stays fixed. We run its small search kernel in eager mode to avoid compilation
# startup time in this CPU example.

# %%
dictionary_cnn, dictionary_history, dictionary_updates, dictionary_validation_seconds = train(dictionary_parameters)
{
    'training seconds': dictionary_history[-1][0],
    'updates': dictionary_updates,
    'validation seconds': dictionary_validation_seconds,
    'best validation error': min(row[2] for row in dictionary_history),
}

# %% [markdown]
# ## Compare training and validation
#
# We plot mean training error over each interval and validation error at its end.
# Training draws new noise each time; validation always uses the same observations.
# The best validation checkpoint, including the initial CNN, is retained for each
# method. Neither run uses the test set to select weights.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show learning curve plotting code"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout='constrained')
for axis, history, title in zip(
    axes, (lbfgs_history, dictionary_history), ('CNN + L-BFGS', 'CNN + dictionary'), strict=True
):
    times, training_errors, validation_errors = zip(*history, strict=True)
    axis.semilogy(times, training_errors, label='Training')
    axis.semilogy(times, validation_errors, '--', label='Validation')
    axis.set(xlabel='Training time (s)', ylabel='Normalized parameter error', title=title)
    axis.legend()
plt.show()

# %% [markdown]
# ## Evaluate on the test phantoms
#
# We evaluate the selected checkpoints on the 16 test phantoms, using the same
# noisy observations for both methods. We report the two trained estimators, using
# `~mr2.operators.functionals.L1Norm` with averaging to compute mean absolute error
# over all test voxels. The normalized parameter error uses
# `~mr2.operators.functionals.MSE`, as in training.
#
# The two estimators solve different fitting problems: L-BFGS includes a quadratic
# regularizer, whereas dictionary matching uses a discrete search and one refinement
# step. These phantoms therefore illustrate training through each estimator, rather
# than establishing their relative performance on measured MRI data.

# %%
test_predictions = {}
test_metrics = {}
for name, cnn, estimate in (
    ('L-BFGS', lbfgs_cnn, lbfgs_parameters),
    ('Dictionary', dictionary_cnn, dictionary_parameters),
):
    m0, t2 = evaluate(cnn, estimate, test_data, test_m0, test_t2)
    test_predictions[name] = (m0, t2)
    test_metrics[name] = {
        'M0 MAE (a.u.)': mr2.operators.functionals.L1Norm(test_m0, divide_by_n=True)(m0)[0].item(),
        'T2 MAE (ms)': 1000 * mr2.operators.functionals.L1Norm(test_t2, divide_by_n=True)(t2)[0].item(),
        'normalized parameter error': parameter_error(m0, t2, test_m0, test_t2).item(),
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
    for name, metrics in test_metrics.items()
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
# We show three test phantoms with different tissue values. All $T_2$ maps use the
# same color scale. The last two columns show the error of each trained estimator.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show test result plotting code"}
fig, axes = plt.subplots(3, 5, figsize=(13, 8), layout='constrained')
for row, index in enumerate((0, 5, 10)):
    maps = (
        test_t2[index],
        test_predictions['L-BFGS'][1][index],
        test_predictions['Dictionary'][1][index],
        test_predictions['L-BFGS'][1][index] - test_t2[index],
        test_predictions['Dictionary'][1][index] - test_t2[index],
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
torch.set_num_threads(previous_threads)
