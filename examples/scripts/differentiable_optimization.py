# %% [markdown]
# # Differentiating through an optimization result
#
# A reconstruction can itself be a layer in a learning problem. Here we solve a small
# regularized inverse problem, then learn its regularization weight from a known target.
# Everything runs on the CPU, with no downloads. We use double precision to make the
# solution and gradient checks easy to inspect.
#
# ## A small inverse problem
#
# We observe $y = Ax_{\mathrm{true}} + \epsilon$, where $A$ attenuates some components.
# For a positive weight $\alpha$, the inner problem is
#
# $$x^*(\alpha) = \arg\min_x \tfrac12\|Ax-y\|_2^2
#                  + \tfrac\alpha2\|x\|_2^2.$$
#
# Its normal equations are $H_\alpha x^* = b$, with
# $H_\alpha = A^H A + \alpha I$ and $b = A^H y$.
# Positive $\alpha$ makes $H_\alpha$ positive definite, so the minimizer is unique.
# A dense solve provides an independent reference; a large MRI problem would use
# operators without explicitly forming this matrix.

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
# ## Factories define the parameterized solve
#
# `~mr2.operators.ConjugateGradientOp` takes two factories. Each receives the tensors
# passed to the solver: here, the weight and observations. One builds the system
# operator, the other returns a **tuple** of right-hand sides. Pass learnable tensors
# explicitly rather than capturing them in a closure: the implicit backward computes
# derivatives with respect to these inputs.
#
# Adding a scalar tensor to a linear operator adds that scalar times the identity.
# Thus ``encoding.gram + alpha`` represents $A^H A + \alpha I$.
# The solver also returns a tuple. We call the instance, rather than ``forward``.


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
# A warm start only helps find the solution: this operator detaches initial values
# requiring gradients, in both modes. It does not learn an initialization.
# For general nonlinear objectives, `~mr2.operators.OptimizerOp` instead takes an
# objective factory and an initializer callable returning a tuple of initial values.
# Its implicit derivative also describes a stationary solution, rather than the path
# from that initialization. The quadratic here needs only CG.
#
# ## Unrolled and implicit derivatives
#
# Our outer loss is $L = \|x^*(\alpha)-x_{\mathrm{true}}\|_2^2/n$.
# Unrolling differentiates the executed CG iterations and keeps their computation
# graph. With early stopping, this is the derivative of that finite computation;
# the stopping decision itself is not differentiated.
#
# Implicit differentiation instead differentiates $H_\alpha x^*=b$. It solves
# $H_\alpha^H z = \nabla_x L$ in the backward pass and gives
# $dL/d\alpha = -z^T x^*$. It avoids storing the forward iteration graph, but requires
# **both** forward and backward solves to converge. The same tolerance and iteration
# limit control both solves in this operator. A small forward residual alone does
# not establish gradient accuracy. Ill conditioning can amplify residual errors.
# The current implicit CG implementation supports first derivatives only.
#
# We compare both modes to autograd through the dense solve and to a centered finite
# difference. We also check the adjoint equation independently with a dense solve.

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
# These values should agree; the dense adjoint residual is a reference check,
# not a measurement of the internal backward CG residual.
{
    'implicit': implicit_gradient.item(),
    'unrolled': unrolled_gradient.item(),
    'dense': reference_gradient.item(),
    'finite difference': finite_difference.item(),
    'forward relative residual': forward_relative_residual.item(),
    'dense adjoint relative residual': backward_relative_residual.item(),
}

# %% [markdown]
# ## Learn the objective parameter
#
# We optimize $\theta = \log\alpha$ so $\alpha = \exp\theta$ stays positive.
# Adam updates only this scalar; each outer step solves the inner reconstruction
# problem again. The target appears only in the outer loss, never in the factories.
# This single synthetic pair demonstrates the mechanism; learning a useful shared
# weight would require separate training and validation data.

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
# The first panel shows the outer loss and the second the parameter being learned.
# The last compares the initial and learned reconstructions with the target.
# Learning reduces error here, but it cannot recover information lost to noise.

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
