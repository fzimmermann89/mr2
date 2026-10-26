import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar, cast

import pytest
import torch
from mr2.operators import functionals
from mr2.operators.Functional import ElementaryFunctional, ElementaryProximableFunctional
from mr2.utils import RandomGenerator
from typing_extensions import TypedDict, Unpack


@dataclass
class FunctionalTestCase:
    """A testcase for proximable functionals used in test parametrization.

    See functional_test_cases.

    """

    functional: ElementaryProximableFunctional
    x_dtype: torch.dtype
    x_shape: torch.Size
    rng: RandomGenerator
    sigma: float | torch.Tensor

    def rand_x(self) -> torch.Tensor:
        """Generate random x"""
        low = 0 if self.x_dtype.is_complex else -2
        return self.rng.rand_tensor(self.x_shape, low=low, high=2.0, dtype=self.x_dtype)

    @property
    def result_dtype(self):
        """Expected result dtype"""
        weight = self.functional.weight
        if isinstance(self.functional, functionals.L21Norm | functionals.NuclearNorm):
            # Radial mappings use the magnitude; a complex weight does not make a real input complex.
            weight = weight.abs()
        return torch.promote_types(self.x_dtype, torch.result_type(self.functional.target, weight))


FUNCTIONALS: list[type[ElementaryFunctional]] = [
    x[1] for x in inspect.getmembers(functionals, lambda x: inspect.isclass(x) and issubclass(x, ElementaryFunctional))
]
PROXIMABLE_FUNCTIONALS: list[type[ElementaryProximableFunctional]] = [
    x for x in FUNCTIONALS if issubclass(x, ElementaryProximableFunctional)
]


Functional = TypeVar('Functional', bound=ElementaryFunctional)


class FunctionalArguments(TypedDict, total=False):
    """Common functional constructor arguments used by the shared tests."""

    target: torch.Tensor | complex | None
    weight: torch.Tensor | complex
    dim: int | Sequence[int] | None
    divide_by_n: bool
    keepdim: bool


def make_functional(functional: type[Functional], **kwargs: Unpack[FunctionalArguments]) -> Functional:
    """Construct a shared test functional with its supported reduction axes."""
    if functional is functionals.L21Norm:
        return cast(Functional, functionals.L21Norm(group_dim=-1, **kwargs))
    if functional is functionals.NuclearNorm:
        # Matrix axes are fixed; the shared dim parameter applies to the other norms.
        kwargs.pop('dim', None)
        return functional(**kwargs)
    return functional(**kwargs)


def functional_test_cases(func: Callable[[FunctionalTestCase], None]) -> Callable[..., None]:
    """Decorator combining multiple parameterizations for test cases for all proximable functionals."""

    @pytest.mark.parametrize('shape', [[1, 2, 3]], ids=['shape=[1,2,3]'])
    @pytest.mark.parametrize('dtype_name', ['float32', 'complex64'])
    @pytest.mark.parametrize('weight', ['scalar_weight', 'tensor_weight', 'complex_weight'])
    @pytest.mark.parametrize('target', ['no_target', 'random_target'])
    @pytest.mark.parametrize('dim', [None], ids=['dim=None'])
    @pytest.mark.parametrize('divide_by_n', [True, False], ids=['mean', 'sum'])
    @pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
    def wrapper(
        functional: type[ElementaryProximableFunctional],
        shape: torch.Size,
        dtype_name: Literal['float32', 'float64', 'complex64', 'complex128'],
        weight: Literal['scalar_weight', 'tensor_weight', 'binary_weight', 'complex_weight'],
        target: Literal['no_target', 'random_target', 'zero_target'],
        dim: Sequence[int] | None,
        divide_by_n: bool,
    ):
        if functional is functionals.NuclearNorm:
            shape = torch.Size((2, 3, 4))
        dtype = getattr(torch, dtype_name)
        rng = RandomGenerator(13)
        sigma = rng.rand_tensor((1,), low=0.0, high=1.0, dtype=dtype.to_real())
        weight_shape = (*shape[:-1], 1) if functional is functionals.L21Norm else shape
        if functional is functionals.NuclearNorm:
            weight_shape = (*shape[:-2], 1, 1)
        match weight:
            case 'scalar_weight':
                weight_value: float | torch.Tensor = rng.float64(low=-2, high=2)
            case 'binary_weight':
                weight_value = rng.rand_tensor(weight_shape, low=0.0, high=1.0, dtype=dtype.to_real()).round()
            case 'complex_weight':
                weight_value = rng.rand_tensor(weight_shape, low=0.0, high=1.0, dtype=dtype.to_complex())
            case 'tensor_weight':
                weight_value = rng.rand_tensor(weight_shape, low=-1.0, high=1.0, dtype=dtype.to_real())
        match target:
            case 'zero_target':
                target_value: torch.Tensor | None = torch.zeros(shape, dtype=dtype)
            case 'no_target':
                target_value = None
            case 'random_target':
                target_value = rng.rand_tensor(shape, low=0 if dtype.is_complex else -1, high=1.0, dtype=dtype)
        f = make_functional(functional, divide_by_n=divide_by_n, weight=weight_value, target=target_value, dim=dim)
        case = FunctionalTestCase(functional=f, x_shape=shape, x_dtype=dtype, rng=rng, sigma=sigma)
        func(case)

    return wrapper
