Operators and reconstruction models
===================================

Start with the high-level reconstruction in :doc:`user_guide`. Use operators
when you need to choose the acquisition model, loss or optimization method.
Operators work on tensors; data containers supply tensors, trajectory and
metadata needed to construct them.

Calling and composing operators
-------------------------------
Call an instance as ``op(x)``, so PyTorch module hooks run. MRtwo operators
return tuples, even for one output. Ordinary neural network modules may
return a tensor; check their call documentation rather than unpacking blindly.

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
``A.H`` is the adjoint, defined by the complex inner product identity; it is
not generally the inverse. Undersampling and density weighting matter.
``A.gram`` represents ``A.H @ A`` and can use an optimized implementation.

Algebra and block operators
---------------------------
``A + B`` sums corresponding outputs for compatible operators. For linear
operators, ``w * A`` weights the output and ``A * w`` weights the input.
A tensor weight must broadcast at the corresponding stage; these expressions
need not agree. Adjoint weights are complex conjugated.

``A | B`` forms a horizontal block row: it takes two inputs and sums
``A(x) + B(z)``. ``A % B`` forms a vertical block column: it takes one input
and returns two outputs. These are tuple blocks, not tensor concatenation.
The adjoint exchanges block rows and columns and sums contributions as needed.

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
precedence (alongside ``@``), then ``+``, then ``|``. Thus an unparenthesized
block expression can build a different model from the one you intended.

A block's mathematical input/output spaces do not label tensor axes. Check
each operator's documented shapes: some act on selected spatial axes, others
add leading parameter or time axes. Broadcasting alone does not establish
that two operators represent compatible physical quantities.

Losses, proximal maps and solvers
---------------------------------
A functional maps tensors to a scalar objective. For example,
``mr2.operators.functionals.MSE(target) @ model_op`` composes a squared-error
loss with a model. A proximable functional additionally supplies a proximal
map for nonsmooth penalties. Its proximal map is not its gradient.

Choose a solver for the objective: conjugate gradient for suitable positive
definite linear systems, gradient methods such as ADAM or LBFGS for smooth
nonlinear fitting, PGD for a smooth term plus a proximable penalty, or PDHG
for supported primal-dual formulations. Check solver assumptions, step sizes,
stopping criteria and tuple input/output conventions in
`mr2.algorithms.optimizers`. An adjoint reconstruction is a useful starting
point, not a substitute for solving every inverse problem.

See :doc:`_notebooks/pgd_wavelet_reconstruction` and
:doc:`_notebooks/tv_minimization_reconstruction_pdhg` for complete objectives.

Autograd and devices
--------------------
Operator composition retains PyTorch's computation graph when the component
operations are differentiable. Set ``requires_grad`` on the tensors or
parameters you want to optimize; avoid detaching or converting to NumPy in
the differentiable path. Iterative algorithms and initialization may have
different differentiation costs and restrictions; consult their documentation.

Move models with ``op.to(device)`` and inputs with ``x.to(device)`` before
calling them. Construct tensor weights and other plain tensor attributes on
the chosen device: module ``to`` moves parameters and registered buffers,
not arbitrary tensor attributes. Use appropriate real/complex dtypes. A dataclass's precision-preserving ``to``
behavior described in :doc:`data` differs from a generic module dtype cast.
