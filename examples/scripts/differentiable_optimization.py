# %% [markdown]
# # Differentiating through an optimization result
#
# We solve a regularized inverse problem and learn its regularization weight by
# differentiating through the solution. We use a synthetic signal with 16 entries,
# so we can compare the iterative solution and its gradient with a dense solve.
# This example runs on the CPU. We use double precision for the numerical checks.
#
# ## A small inverse problem
#
# We simulate measurements $y = Ax_{\mathrm{true}} + \epsilon$ with additive Gaussian
# noise. The singular values of $A$ range from 0.15 to 1, so directly inverting $A$
# amplifies noise in components with small singular values. To regularize the
# reconstruction, we minimize
#
# $$x^*(\alpha) = \arg\min_x \tfrac12\|Ax-y\|_2^2
#                  + \tfrac\alpha2\|x\|_2^2.$$
#
# Its normal equations are $H_\alpha x^* = b$, with
# $H_\alpha = A^H A + \alpha I$ and $b = A^H y$.
# For $\alpha > 0$, $H_\alpha$ is positive definite and the minimizer is unique.
# We solve these equations using conjugate gradient (CG) and compare with
# ``torch.linalg.solve``. For large MRI problems, we can apply the normal operator
# without forming its matrix.

# %%
import matplotlib.pyplot as plt
import torch
from mr2.operators import ConjugateGradientOp, EinsumOp, LinearOperator

rng = torch.Generator(device='cpu').manual_seed(7)
dtype = torch.float64
size = 16
coordinate = torch.linspace(0, 1, size, dtype=dtype)
truth = torch.sin(2 * torch.pi * coordinate) + 0.4 * torch.cos(6 * torch.pi * coordinate)
basis, _ = torch.linalg.qr(torch.randn(size, size, generator=rng, dtype=dtype))
singular_values = torch.linspace(0.15, 1, size, dtype=dtype)
matrix = torch.diag(singular_values) @ basis.T
encoding = EinsumOp(matrix, 'i j, j -> i')
(clean_data,) = encoding(truth)
data = clean_data + 0.12 * torch.randn(size, generator=rng, dtype=dtype)

# Operators return tuples, even for one input and one output. Unpack with (value,).
# @ composes operators; .H is the adjoint; .gram represents A^H A.
(explicit_gram_truth,) = (encoding.H @ encoding)(truth)
(gram_truth,) = encoding.gram(truth)
torch.testing.assert_close(gram_truth, explicit_gram_truth)
normal_matrix = matrix.T @ matrix
(rhs,) = encoding.H(data)
identity = torch.eye(size, dtype=dtype)

# %% [markdown]
# ## Set up the CG solver
#
# We use `~mr2.operators.ConjugateGradientOp` with two factory functions. Both take
# the regularization weight and observations as inputs. The first returns
# $H_\alpha$; the second returns the right-hand side as a tuple.
# We pass the learnable weight as a solver input so that the implicit backward
# can compute its derivative.
#
# Adding a scalar tensor to a linear operator adds that scalar times the identity.
# Thus ``encoding.gram + alpha`` represents $A^H A + \alpha I$.
# Like other MRtwo operators, the solver returns a tuple, which we unpack below.


# %%
def operator_factory(alpha: torch.Tensor, _observations: torch.Tensor) -> LinearOperator:
    """Build the regularized normal operator."""
    return encoding.gram + alpha


def rhs_factory(_alpha: torch.Tensor, observations: torch.Tensor) -> tuple[torch.Tensor]:
    """Build the right-hand side from the observations."""
    return encoding.H(observations)


implicit_solver = ConjugateGradientOp(operator_factory, rhs_factory, tolerance=1e-12, max_iterations=64)
unrolled_solver = ConjugateGradientOp(
    operator_factory, rhs_factory, implicit_backward=False, tolerance=1e-12, max_iterations=64
)
alpha_check = torch.tensor(0.2, dtype=dtype, requires_grad=True)
# initial_value is a tuple containing a constant warm start; None starts from zero.
(solution,) = implicit_solver(alpha_check, data, initial_value=(torch.zeros_like(truth),))
reference = torch.linalg.solve(normal_matrix + alpha_check * identity, rhs)
torch.testing.assert_close(solution, reference, rtol=1e-9, atol=1e-10)
(system_solution,) = operator_factory(alpha_check, data)(solution)
forward_relative_residual = (system_solution - rhs).norm() / rhs.norm()
assert forward_relative_residual < 1e-10

# %% [markdown]
# The optional ``initial_value`` gives CG a constant warm start. The solver detaches
# initial values that require gradients in both differentiation modes.
# For nonlinear objectives, we can use `~mr2.operators.OptimizerOp` with an objective
# factory and an initializer that returns a tuple of initial values. Its implicit
# backward requires a stationary solution with an invertible Hessian; its CG
# backward solve also requires a positive definite Hessian. Here we use the normal
# equations directly.
#
# ## Unrolled and implicit derivatives
#
# We measure reconstruction error with the outer loss
# $L = \|x^*(\alpha)-x_{\mathrm{true}}\|_2^2/n$, where $n=16$.
# Unrolling differentiates the executed CG iterations and keeps their computation
# graph. With early stopping, this is the derivative of that finite computation;
# the stopping decision itself is not differentiated.
#
# Implicit differentiation instead differentiates $H_\alpha x^*=b$. It solves
# $H_\alpha^H z = \nabla_x L$ in the backward pass and gives
# $dL/d\alpha = -z^T x^*$. It avoids storing the forward iteration graph, but requires
# both forward and backward solves to converge. The same tolerance and iteration
# limit control both solves in this operator. A small forward residual alone does
# not establish gradient accuracy. Ill conditioning can amplify residual errors.
# The current implicit CG implementation supports first derivatives only.
#
# We compare the gradients with autograd through the dense solve and a centered
# finite difference. We also solve the adjoint equation with ``torch.linalg.solve``
# and evaluate $-z^T x^*$. The dense adjoint residual checks this reference solve;
# it does not measure the internal backward CG residual.

# %%
implicit_loss = (solution - truth).square().mean()
(implicit_gradient,) = torch.autograd.grad(implicit_loss, alpha_check)
(unrolled_solution,) = unrolled_solver(alpha_check, data)
unrolled_loss = (unrolled_solution - truth).square().mean()
(unrolled_gradient,) = torch.autograd.grad(unrolled_loss, alpha_check)
reference_loss = (reference - truth).square().mean()
(reference_gradient,) = torch.autograd.grad(reference_loss, alpha_check)
step = 1e-5
with torch.no_grad():
    (plus,) = implicit_solver(alpha_check + step, data)
    (minus,) = implicit_solver(alpha_check - step, data)
    finite_difference = ((plus - truth).square().mean() - (minus - truth).square().mean()) / (2 * step)
    hessian = normal_matrix + alpha_check * identity
    loss_gradient = 2 * (solution - truth) / size
    adjoint_solution = torch.linalg.solve(hessian.T, loss_gradient)
    backward_relative_residual = (hessian.T @ adjoint_solution - loss_gradient).norm() / loss_gradient.norm()
    analytic_gradient = -(adjoint_solution * solution).sum()

for gradient in (implicit_gradient, unrolled_gradient, finite_difference, analytic_gradient):
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-6, atol=1e-9)
assert backward_relative_residual < 1e-10
{
    'implicit': implicit_gradient.item(),
    'unrolled': unrolled_gradient.item(),
    'dense': reference_gradient.item(),
    'finite difference': finite_difference.item(),
    'forward relative residual': forward_relative_residual.item(),
    'dense adjoint relative residual': backward_relative_residual.item(),
}

# %% [markdown]
# ## Learn the regularization weight
#
# We optimize $\theta = \log\alpha$ so $\alpha = \exp\theta$ stays positive.
# At each Adam step, we reconstruct the signal and backpropagate the outer loss
# through the CG solution. We use the target only to evaluate this loss.
# We fit the weight to one noisy signal here. To assess whether a learned weight
# generalizes, we would need separate training and validation data.

# %%
log_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
outer_optimizer = torch.optim.Adam([log_alpha], lr=0.08)
loss_history = []
alpha_history = []
for _ in range(120):
    outer_optimizer.zero_grad()
    alpha = log_alpha.exp()
    (reconstruction,) = implicit_solver(alpha, data)
    outer_loss = (reconstruction - truth).square().mean()
    loss_history.append(outer_loss.item())
    alpha_history.append(alpha.item())
    outer_loss.backward()
    outer_optimizer.step()

with torch.no_grad():
    learned_alpha = log_alpha.exp()
    (learned_solution,) = implicit_solver(learned_alpha, data)
    learned_loss = (learned_solution - truth).square().mean()
    (initial_solution,) = implicit_solver(torch.tensor(1.0, dtype=dtype), data)
    learned_reference = torch.linalg.solve(normal_matrix + learned_alpha * identity, rhs)
torch.testing.assert_close(learned_solution, learned_reference, rtol=1e-9, atol=1e-10)
assert torch.isfinite(learned_loss)
assert learned_loss < 0.5 * loss_history[0]
{'initial MSE': loss_history[0], 'learned MSE': learned_loss.item(), 'learned alpha': learned_alpha.item()}

# %% [markdown]
# We plot the loss and regularization weight over the outer iterations, then
# compare the reconstructions. The learned weight reduces the error for this signal.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting code"}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
axes[0].plot(loss_history)
axes[0].set(xlabel='Outer step', ylabel='Mean squared error', title='Outer loss')
axes[1].semilogy(alpha_history)
axes[1].set(xlabel='Outer step', ylabel=r'$\alpha$', title='Regularization weight')
axes[2].plot(coordinate, truth, 'k.-', label='Target')
axes[2].plot(coordinate, initial_solution, '--', label='Initial weight')
axes[2].plot(coordinate, learned_solution, '.-', label='Learned weight')
axes[2].set(xlabel='Coordinate', ylabel='Signal', title='Reconstruction')
axes[2].legend()
fig.tight_layout()
plt.show()
