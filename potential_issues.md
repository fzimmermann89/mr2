# Potential Issues

This document contains only issues retained after a second source audit and
targeted runtime checks. It excludes design preferences, expected validation
delegated to PyTorch/einops, and earlier claims that did not reproduce.

## Summary

| Severity | Issue | Validation |
| --- | --- | --- |
| High | `KData.from_file()` filters sample data without filtering associated acquisition state | Confirmed by control flow |
| High | `DictionaryMatchOp` allows zero-norm atoms to silently win matches | Reproduced |
| Medium | `MultiHeadAttention` cross-attention fails when cross features differ from input features | Reproduced |
| Medium | Three attention layers accept non-divisible channel/head configurations that cannot execute | Reproduced |
| Medium | Convolutional dictionary operators accept even kernels but fail in forward application | Reproduced |
| Medium | `LeWinTransformerBlock` fails for non-window-aligned image sizes | Reproduced |
| Medium | PDHG automatic steps produce `NaN` output for a valid zero operator | Reproduced |
| Low | `DropPath(1.0, scale_by_keep=True)` produces `NaN` output | Reproduced |
| Low | `KTrajectory.from_ismrmrd()` silently discards trajectory coordinates beyond three | Confirmed by control flow |
| Low | `KData.reshape_by_idx()` warning is malformed | Confirmed by source |

## High Severity

### 1. `KData.from_file()` can construct inconsistent state after dropping variable-length acquisitions

**Location:** `src/mr2/data/KData.py:172-269`

**Problem**

`from_file()` calculates per-acquisition effective readout lengths, warns when
they differ, and stacks only entries whose length matches `shapes[-1]`:

```python
data = torch.stack(
    [
        ...
        if acq.data.shape[-1] - pre - post == shapes[-1]
    ]
)
```

However, the selection is not applied to:

- `acquisitions`, later used by `KTrajectoryIsmrmrd`;
- `acq_info`, used to create the header and sort/reshape the result;
- `k0_center` and `n_k0_tensor`, used by calculated trajectories;
- the reverse-readout and phase-encoding indices derived from the full header.

**Impact**

The branch announces that incompatible acquisitions are being discarded, but the
resulting metadata and trajectory calculation still refer to the original
acquisition count. Data with variable effective readout lengths can therefore
fail after import in trajectory calculation or `reshape_by_idx()`. This is a
data-loading failure on an input pattern the function explicitly attempts to
handle.

**Validation**

Confirmed from control flow: the filtering conditional exists only in the data
stack at lines 207-213. All later consumers use state constructed before that
conditional. Existing `KData` tests cover differing phase-encoding counts and
discard padding that reduces back to a uniform effective length, but not an
actual variable effective readout-length case.

**Potential fix**

1. Build a boolean `keep` mask from effective sample counts before constructing
   dependent state.
2. Apply the mask consistently to `acquisitions`, `acq_info`, `k0_center`,
   `n_k0_tensor`, `discard_pre`, and `discard_post`.
3. Construct `KHeader` and all trajectory variants only from retained entries.
4. Consider making the retained length policy explicit. The current use of
   `shapes[-1]` selects the largest unique effective length, not necessarily the
   most common acquisition length.

**Regression tests**

- Import a file containing two effective readout lengths using
  `DummyTrajectory()` or another `KTrajectoryCalculator`; verify a consistent
  returned data/header/trajectory acquisition count.
- Repeat the case with `KTrajectoryIsmrmrd()`.
- Verify retained acquisition indices correspond to the retained data entries.

### 2. `DictionaryMatchOp` can silently return the wrong parameter for a zero-norm dictionary atom

**Location:** `src/mr2/operators/DictionaryMatchOp.py:85-110, 151-168`

**Problem**

`append()` normalizes generated dictionary columns without checking their norm:

```python
inverse_norm_y = torch.linalg.norm(y, dim=0).reciprocal()
y = y * inverse_norm_y
```

A generated all-zero signal becomes a `NaN` column. During matching, similarity
values for that column remain `NaN`; `argmax()` can then choose it over a finite,
exact match.

**Impact**

This is silent result corruption rather than an early error. For quantitative
dictionary matching, a degenerate model sample can cause an unrelated input to
receive the wrong parameter estimate.

**Validation**

Reproduced with a two-entry model yielding dictionary signals `[0, 0]` and
`[1, 1]`. After appending, `torch.isfinite(operator.y).all()` is `False`, and
matching input `[1, 1]` returns the zero-signal entry instead of the exact
non-zero entry. Current tests exercise ordinary model entries but do not call
matching with an appended zero-norm signal.

**Potential fix**

The clearest default is to fail in `append()`:

1. Calculate norms before reciprocal normalization.
2. Detect `norm <= eps` for the signal dtype.
3. Raise `ValueError` identifying how many dictionary entries are degenerate and
   that normalized matching is undefined for them.

If zero atoms must be supported, exclude them from candidate matching and define
the behavior for an all-zero input explicitly. Silently replacing the norm with
epsilon would keep a meaningless atom in the search and is less safe.

**Regression tests**

- Verify appending a zero-norm atom raises a targeted `ValueError`, or verify the
  explicitly chosen exclusion behavior.
- Verify a valid atom remains selectable when a degenerate model evaluation is
  encountered.
- Cover both real and complex dictionary signals.

## Medium Severity

### 3. `MultiHeadAttention` cross-attention fails when `n_channels_cross != n_channels_in`

**Location:** `src/mr2/nn/attention/MultiHeadAttention.py:42-55, 88-101`

**Problem**

The public constructor exposes `n_channels_cross` for cross-attention. The
implementation derives the query attention width from `n_channels_in`, but
derives key/value attention width from `n_channels_cross`:

```python
channels_per_head_q = n_channels_in // n_heads
channels_per_head_kv = n_channels_kv // n_heads
```

Scaled dot-product attention requires the projected query and key feature widths
to match; input feature widths may differ, but projected attention widths may not.

**Impact**

The advertised cross-attention configuration fails at runtime for ordinary use
where context embeddings have a different channel width from image/features.

**Validation**

- `MultiHeadAttention(4, 4, 2, n_channels_cross=4)` executes successfully.
- `MultiHeadAttention(4, 4, 2, n_channels_cross=6)` raises an internal
  scaled-dot-product-attention size mismatch.
- No targeted test for `MultiHeadAttention` cross-attention was found.

**Potential fix**

Use one attention embedding width for query/key/value and treat
`n_channels_cross` only as the input width of the key/value projection. For the
current shape-preserving design, after validating query divisibility:

```python
attention_width = n_channels_in
self.to_q = Linear(n_channels_in, attention_width)
self.to_kv = Linear(n_channels_cross, 2 * attention_width)
self.to_out = Linear(attention_width, n_channels_out)
```

Alternatively, add an explicit `attention_width` parameter and project all
three streams to it.

**Regression tests**

- Cross-attention with unequal query and context input channels.
- Features-first and `features_last=True` layouts.
- Cross-attention combined with non-default `n_channels_out`.

### 4. Attention constructors do not reject channel/head settings that cannot execute

**Location:** `src/mr2/nn/attention/MultiHeadAttention.py:48-55, 88-101`;
`src/mr2/nn/attention/ShiftedWindowAttention.py:59-65, 104-131`;
`src/mr2/nn/attention/TransposedAttention.py:37-49`

**Problem**

Each layer computes `n_channels_in // n_heads` without asserting that the
division is exact. In these three implementations, the rounded-down projection
size conflicts with a later layer or grouped-convolution constraint.

**Impact**

Misconfigured networks fail with low-level matrix multiplication or convolution
errors rather than a constructor error explaining the invalid model
configuration. In `TransposedAttention`, the failure occurs during construction;
in the other modules it occurs on the first input batch.

**Validation**

| Configuration | Result |
| --- | --- |
| `MultiHeadAttention(5, 5, 2)` | Fails in `to_out`: attention width is 4 but layer expects 5 |
| `ShiftedWindowAttention(2, 5, 5, 2)` | Fails in `to_qkv`: layer expects 4 input features but receives 5 |
| `TransposedAttention(2, 5, 5, 2)` | Fails constructing grouped convolution |
| `TransposedAttention(2, 4, 4, 2)` | Executes successfully |

The existing shifted-window and transposed-attention tests use divisible widths
only.

**Potential fix**

Add constructor validation to all three classes:

```python
if n_channels_in % n_heads != 0:
    raise ValueError('n_channels_in must be divisible by n_heads.')
```

For `MultiHeadAttention`, coordinate this with the cross-attention fix above:
after key/value inputs project to the query attention width,
`n_channels_cross` need not itself be divisible by `n_heads`.

**Regression tests**

- Parameterized constructor tests asserting a clear `ValueError` for
  incompatible `(channels, heads)` pairs.
- A valid non-equal cross-attention input-width test for `MultiHeadAttention`, so
  validation does not accidentally reject supported context widths.

### 5. Convolutional dictionary operators accept even kernels but cannot preserve output shape

**Location:** `src/mr2/operators/ConvAnalysisDictionaryOp.py:42-47, 93-103`;
`src/mr2/operators/ConvSynthesisDictionaryOp.py:40-45, 93-102`

**Problem**

Both operators document shape-preserving application and describe odd kernels as
typical, not required. Padding is computed symmetrically as `(k // 2, k // 2)`
for every kernel dimension. For an even kernel, total padding is `k`, while a
shape-preserving valid convolution needs total padding `k - 1`. The convolution
therefore returns one extra spatial element and the subsequent reshape fails.

**Impact**

A documented constructor input is accepted and then fails on use. This affects
both analysis and synthesis operators and prevents use of otherwise valid even
filter banks.

**Validation**

- `(1, 3, 3)` kernel on a `4 x 4` analysis input succeeds with output shape
  `(1, 4, 4)`.
- `(1, 2, 2)` kernel on the same input raises an invalid reshape error because
  convolution produced `5 x 5` values.
- `ConvSynthesisDictionaryOp` fails analogously for `(1, 2, 2)`.
- Existing tests parameterize only odd spatial kernel sizes.

**Potential fix**

Choose one supported contract:

- If even filters are not intended, validate all spatial kernel extents as odd
  in both constructors and update the documentation.
- If they are intended, calculate asymmetric padding whose left/right sum is
  `k - 1`, and ensure the adjoint uses exactly the corresponding adjoint padding
  operation. This route needs adjointness tests for even kernels and each
  `pad_mode`.

**Regression tests**

- Constructor-rejection tests for even extents, or forward/adjoint and
  dot-product-adjointness tests for even 1D, 2D, and mixed odd/even kernels.

### 6. `LeWinTransformerBlock` contradicts window-attention padding support for unaligned image sizes

**Location:** `src/mr2/nn/nets/Uformer.py:79-105`;
`src/mr2/nn/attention/ShiftedWindowAttention.py:98-126`

**Problem**

`ShiftedWindowAttention` explicitly pads inputs whose spatial sizes are not
multiples of `window_size`, then crops the result. `LeWinTransformerBlock`
precedes it by adding a learned modulator generated using floor-based tiling:

```python
modulator = self.modulator.tile(
    [t // s for t, s in zip(x.shape[1:], self.modulator.shape, strict=False)]
)
```

For a remainder pixel, the modulator is smaller than `x`, so execution fails
before window attention gets a chance to apply its padding support.

**Impact**

`Uformer` cannot process common odd or otherwise unaligned spatial sizes even
though its attention component supports them. This can surface only at inference
on real image dimensions after training with aligned crops.

**Validation**

- `LeWinTransformerBlock(..., window_size=2)` executes for input shape
  `(1, 4, 16, 16)`.
- The same block raises a tensor-size mismatch for `(1, 4, 15, 16)`.
- Existing `Uformer` tests use aligned size `16` only; the standalone
  `ShiftedWindowAttention` tests already prove that layer supports remainders.

**Potential fix**

Use ceil-based repetition for `self.modulator` and crop it to `x.shape[1:]`
before addition. This preserves the periodic-modulator interpretation while
matching the padding behavior already supported by attention. Alternatively,
explicitly require each input/stage resolution to be divisible by `window_size`,
but that would intentionally remove capability currently present in the lower
level attention module.

**Regression tests**

- `LeWinTransformerBlock` and end-to-end `Uformer` forward tests on spatial
  shapes with a remainder in one and multiple dimensions.
- Include dimensions whose downsampled U-Net stages also have remainders.

### 7. PDHG automatic step-size selection returns `NaN` for a valid zero operator

**Location:** `src/mr2/algorithms/optimizers/pdhg.py:174-212`

**Problem**

If either step size is omitted, the implementation estimates `operator_norm` and
divides by it. For `K = 0`, its norm is zero and the problem
`min_x f(0) + g(x)` remains valid, but the automatic step path has no zero-norm
case.

**Impact**

The optimizer returns non-finite values for a well-defined degenerate problem.
Zero operators can arise directly or from operator algebra/configuration in
regularized reconstruction workflows.

**Validation**

- `pdhg(None, L2NormSquared(), ZeroOp(keep_shape=True), (tensor([2.0]),),
  max_iterations=1)` with automatic steps returns non-finite output.
- Passing `primal_stepsize=1.0` and `dual_stepsize=1.0` for the same problem
  returns finite output.
- Existing PDHG auto-step tests use non-zero identity-based operators.

**Potential fix**

After estimating the norm, check that it is finite and strictly positive before
division. For a zero norm, either:

- choose finite fallback step sizes because the coupling constraint is vacuous
  when `K = 0`; or
- raise a targeted `ValueError` requesting explicit step sizes.

The fallback option provides more useful behavior for valid decoupled problems,
but it should be documented.

**Regression tests**

- Zero-operator automatic-step execution with both `f` and `g` variations.
- Non-finite or zero norm estimates produce defined behavior.
- Existing positive-norm automatic-step behavior remains unchanged.

## Low Severity

### 8. `DropPath(1.0, scale_by_keep=True)` creates `NaN` activations

**Location:** `src/mr2/nn/DropPath.py:19-31, 47-55`

**Problem**

`droprate=1.0` is already supported for the unscaled path, but when
`scale_by_keep=True`, line 54 divides the zero keep mask by `1 - droprate = 0`.

**Impact**

A boundary-value module configuration contaminates activations with `NaN` values.
The failure is localized and easy to trigger, hence low rather than medium
severity.

**Validation**

- `DropPath(1.0, scale_by_keep=False)` is covered by an existing test and returns
  zeros.
- `DropPath(1.0, scale_by_keep=True)(torch.ones(3, 4))` returns a tensor for
  which `torch.isfinite(...).all()` is `False`.

**Potential fix**

Validate `0 <= droprate <= 1`. For the full-drop scaled case, either return
zeros before scaling or reject `droprate == 1 and scale_by_keep` because
expectation-preserving rescaling is undefined when no sample can be kept.

**Regression tests**

- Boundary tests for `droprate=0`, `droprate=1`, and invalid values outside the
  allowed interval, with both `scale_by_keep` settings.

### 9. `KTrajectory.from_ismrmrd()` accepts and discards trajectory coordinates beyond three

**Location:** `src/mr2/data/KTrajectory.py:109-110, 168-195`

**Problem**

The method intends to enforce a 3D trajectory. It pads one- and two-coordinate
inputs, but for inputs whose last dimension is greater than three the expression
`[zero] * (3 - traj.shape[-1])` contributes no tensors. The unchanged trajectory
is then passed to `from_tensor()`, which unbinds all coordinates and selects only
the first `x`, `y`, and `z` coordinates.

**Impact**

If an ISMRMRD source carries extra trajectory coordinates, information is lost
without warning. This may hide an unsupported file convention rather than fail
clearly at import time.

**Validation**

Confirmed by direct control flow: no `> 3` branch exists and `from_tensor()`
indexes only coordinates in `axes_order='xyz'`. Existing trajectory file tests
use two-coordinate trajectories and do not cover extra components.

**Potential fix**

Replace the single `!= 3` branch with explicit cases:

- `1` or `2`: pad missing spatial coordinates with zeros;
- `3`: accept unchanged;
- greater than `3`: raise `ValueError` describing unsupported trajectory
  dimensionality.

Also decide explicitly how a zero-coordinate ISMRMRD trajectory should be
handled rather than allowing a downstream indexing error.

**Regression tests**

- Inputs with one, two, three, four, and zero trajectory coordinates.
- Confirm the error includes the received coordinate count for unsupported files.

### 10. `KData.reshape_by_idx()` emits a malformed warning for irregular label combinations

**Location:** `src/mr2/data/KData.py:316-325`

**Problem**

Two adjacent string literals are malformed in the fallback warning:

```python
f'There are different numbers of acquisistions in'
'different combinations of labels {"/".join(OTHER_LABELS)}: \n'
```

The emitted message contains `indifferent` without a separating space and prints
the expression `{"/".join(OTHER_LABELS)}` literally instead of the relevant
label names.

**Impact**

This does not change numerical results, but it degrades the diagnostic intended
to tell users how to repair an irregular acquisition layout.

**Validation**

Confirmed directly from the adjacent non-formatted string literals. The existing
irregular-repetition test checks only that a warning containing "different
number" occurs, not the full diagnostic text.

**Potential fix**

Use a single formatted message, correct the spelling in the same change, and
include the labels:

```python
f'There are different numbers of acquisitions in different combinations '
f'of labels {"/".join(OTHER_LABELS)}:\n'
```

**Regression tests**

- Assert the fallback warning names the expected labels and contains readable
  wording for an irregular-repetition fixture.

## Removed From The Earlier Review

The following classes of claims were removed because they did not demonstrate a
defect on reinspection:

- `Add`, `ZeroOp`, and `RandomGenerator` scalar/dtype claims did not reproduce.
- Shape-repair joins, FiLM omission behavior, and packaging/import breadth are
  intentional API or maintenance tradeoffs rather than correctness findings.
- `LinearSelfAttention` and `NeighborhoodSelfAttention` can project through a
  reduced internal width; the non-divisibility failure reproduced only in the
  three attention implementations listed above.
- `TransposedAttention` grouped convolution is valid for divisible channels; its
  retained issue is the absent divisibility guard, not an always-invalid design.
