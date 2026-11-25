Operators and reconstruction models
===================================

Use operators to define an acquisition model or change a reconstruction's
loss and solver. The high-level reconstruction in :doc:`user_guide` combines
these components for you. Operators act on tensors; construct them using the
trajectory and metadata from a data container.

Calling and composing operators
-------------------------------
Call an instance as ``op(x)``, so PyTorch module hooks run. MRtwo operators
return tuples, even for one output. Ordinary neural network modules may
return a tensor. Check the return type before unpacking a network's output.

.. code-block:: python

    import torch
    import mr2

    x = torch.randn(1, 1, 1, 8, 8, dtype=torch.complex64, requires_grad=True)
    identity_op = mr2.operators.IdentityOp()
    scaled_op = 2 * identity_op
    model_op = scaled_op @ identity_op
    (y,) = model_op(x)
    (backprojected,) = model_op.H(y)
    (normal_result,) = model_op.gram(x)
    torch.testing.assert_close(normal_result, 4 * x)
    y.abs().square().sum().backward()

``A @ B`` applies ``B`` first, then passes its output tuple as inputs to ``A``.
For MRI, ``fourier_op @ sensitivity_op`` maps an image to coil k-space.
``A.H`` is the adjoint, defined by the complex inner product identity.
It generally differs from the inverse: applying an undersampled Fourier
operator's adjoint backprojects the measured samples without recovering the
missing measurements. ``A.gram`` computes ``A.H @ A`` and can use an optimized
implementation.

Algebra and block operators
---------------------------
``A + B`` sums corresponding outputs for compatible operators. For linear
operators, ``w * A`` weights the output and ``A * w`` weights the input.
A tensor weight must broadcast at the corresponding stage; these expressions
need not agree. Adjoint weights are complex conjugated.

``A | B`` forms a horizontal block row: it takes two inputs and sums
``A(x) + B(z)``. ``A % B`` forms a vertical block column: it takes one input
and returns two outputs as a tuple. Block stacking keeps the tensors separate.
The adjoint exchanges block rows and columns and sums their contributions.

.. code-block:: python

    row_op = identity_op | scaled_op
    (combined,) = row_op(x, x)
    column_op = identity_op % scaled_op
    first, second = column_op(x)
    block_op = (identity_op | scaled_op) % (scaled_op | identity_op)
    top, bottom = block_op(x, x)
    torch.testing.assert_close(combined, 3 * x)
    torch.testing.assert_close(top, bottom)

Parenthesize block expressions. Python gives ``%`` the multiplication-level
precedence (alongside ``@``), above ``+`` and ``|``. Use parentheses to make
the intended rows and columns explicit.

Check tensor shapes within each block. Some operators act on selected
spatial axes; others add leading parameter or time axes. Even if tensors
broadcast, their axes and physical units must agree with the model.

Losses, proximal maps and solvers
---------------------------------
A functional maps tensors to a scalar objective. For example,
``mr2.operators.functionals.MSE(target) @ model_op`` composes a squared-error
loss with a model. A proximable functional supplies a proximal map, which
lets splitting methods handle nonsmooth penalties. A proximal map and a
gradient perform different updates.

Choose a solver that matches the objective. Conjugate gradient solves
suitable positive definite linear systems; ADAM and LBFGS support smooth
nonlinear fitting. Use PGD for a smooth term plus a proximable penalty, or
PDHG for supported primal-dual formulations. Check the assumptions and step
sizes in `mr2.algorithms.optimizers`, along with the stopping criteria and
tuple input/output conventions. You can use an adjoint reconstruction as an
initial estimate for an iterative solver.

See :doc:`_notebooks/pgd_wavelet_reconstruction` and
:doc:`_notebooks/tv_minimization_reconstruction_pdhg` for complete objectives.

Autograd and devices
--------------------
Operator composition retains PyTorch's computation graph when the component
operations are differentiable. Set ``requires_grad`` on the tensors or
parameters you want to optimize, and keep them in the graph by avoiding
``detach`` and NumPy conversions along that path. Check each iterative
algorithm's differentiation support and memory requirements before using it
inside a trainable model.

Move models with ``op.to(device)`` and inputs with ``x.to(device)`` before
calling them. Construct tensor weights and other plain tensor attributes on
the chosen device: module ``to`` moves parameters and registered buffers,
not arbitrary tensor attributes. Use compatible real/complex dtypes. The
dataclass ``to`` method preserves tensor kinds as described in :doc:`data`;
a generic module dtype cast follows PyTorch's conversion rules.
