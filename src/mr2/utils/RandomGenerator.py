"""Random generator."""

from collections.abc import Sequence
from math import ceil, floor

import torch


def check_bounds(low: float | int | torch.Tensor, high: float | int | torch.Tensor, dtype: torch.dtype | None) -> None:
    """Clip the bounds to a range matching the dtype.

    Parameters
    ----------
    low
        Lower bound.
    high
        Upper bound.
    dtype
        Data type, used to find allowed range.
    """
    info: torch.finfo | torch.iinfo
    if dtype is None:
        info = torch.finfo()
        minval, maxval = info.min, info.max
    elif dtype.is_floating_point:
        info = torch.finfo(dtype)
        minval, maxval = info.min, info.max
    else:
        info = torch.iinfo(dtype)
        minval = info.min
        if dtype in (torch.int64, torch.uint64):
            maxval = info.max  # https://github.com/pytorch/pytorch/issues/81446
        else:
            maxval = info.max + 1
    if low > high:
        raise ValueError('low should be lower than high')
    if low < minval or high > maxval:
        raise ValueError(f'low/high should be in the range of {info.min} and {info.max} for {dtype}')


class RandomGenerator:
    """Generate random numbers for various purposes.

    Uses a fixed seed to ensure reproducibility.

    Provides:
        - Scalar uniform random numbers:
            int8, int16, int32, int64, uint8, uint16, uint32, uint64,
            float32, float64, complex64, complex128
        - Tensor of uniform random numbers:
            int8_tensor, int16_tensor, int32_tensor, int64_tensor, uint8_tensor,
            float32_tensor, float64_tensor, complex64_tensor, complex128_tensor
            (Note: uint16, uint32, uint64 tensors are not yet supported by PyTorch)
        - Tuple of uniform random numbers:
            int8_tuple, int16_tuple, int32_tuple, int64_tuple,
            uint8_tuple, uint16_tuple, uint32_tuple, uint64_tuple,
            float32_tuple, float64_tuple, complex64_tuple, complex128_tuple
    """

    def __init__(self, seed: int | None = None, device: torch.device | str | None = None):
        """Initialize the random generator with a fixed seed.

        Parameters
        ----------
        seed
            Seed for the random generator. If `None`, use
            default generator to get a random seed
        device
            Device for the underlying torch generator. If `None`, the default device is used.
        """
        seed_ = int(torch.randint(0, 2**32, (1,))) if seed is None else seed
        self.generator = torch.Generator(device=device).manual_seed(seed_)

    def _randint(
        self,
        size: Sequence[int] | int,
        low: int,
        high: int,
        dtype: torch.dtype = torch.int64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generate a tensor of uniformly distributed random integers in [low, high).
        
        Parameters:
            size (int or Sequence[int]): Shape of the output tensor.
            low (int): Inclusive lower bound of the sampled integers.
            high (int): Exclusive upper bound of the sampled integers.
            dtype (torch.dtype): Desired integer dtype of the result.
            device (torch.device or str or None): If provided, the result is moved to this device; otherwise the tensor is produced on the generator's device.
        
        Returns:
            torch.Tensor: Tensor of shape `size` containing integers in the interval [low, high) with the requested `dtype`.
        """
        check_bounds(low, high, dtype)
        size_ = (size,) if isinstance(size, int) else size
        tensor = torch.randint(low, high, size_, generator=self.generator, dtype=dtype, device=self.generator.device)
        return tensor.to(device=device) if device is not None else tensor

    def _rand(
        self,
        size: Sequence[int] | int,
        low: float | torch.Tensor,
        high: float | torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generate a tensor of uniform random values in [low, high).
        
        Parameters:
            size (int or Sequence[int]): Shape of the output tensor.
            low (float or torch.Tensor): Lower bound (inclusive) of the sampling interval.
            high (float or torch.Tensor): Upper bound (exclusive) of the sampling interval.
            dtype (torch.dtype): Data type of the output tensor.
            device (torch.device or str or None): Device for the returned tensor. If `None`, the tensor is created on the internal generator's device; otherwise the result is moved to `device`.
        
        Returns:
            torch.Tensor: Tensor of shape `size` and dtype `dtype` containing samples drawn uniformly from [low, high).
        
        Raises:
            ValueError: If bounds are invalid (e.g., low > high or bounds outside the representable range for `dtype`).
        """
        check_bounds(low, high, dtype)
        tensor = (
            torch.rand(size, generator=self.generator, dtype=dtype, device=self.generator.device) * (high - low)
        ) + low
        return tensor.to(device=device) if device is not None else tensor

    def float32_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: float = 0.0,
        high: float = 1.0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate a float32 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound.
        high
            Upper bound.
        device
            Device of the output tensor. If `None`, the tensor is created on the generator device.

        Returns
        -------
            Tensor of float32 random numbers.
        """
        return self._rand(size, low, high, torch.float32, device=device)

    def float64_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: float = 0.0,
        high: float = 1.0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generate a tensor of uniformly distributed float64 values in [low, high).
        
        Parameters:
            size (Sequence[int] | int): Shape of the output tensor.
            low (float): Lower bound (inclusive).
            high (float): Upper bound (exclusive).
            device (torch.device | str | None): Device for the output tensor; if `None`, the generator's device is used.
        
        Returns:
            torch.Tensor: Tensor of dtype torch.float64 and shape `size` with values sampled uniformly from [low, high).
        """
        return self._rand(size, low, high, torch.float64, device=device)

    def complex64_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: float = 0.0,
        high: float = 1.0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generate a complex64 tensor whose amplitudes are uniformly sampled in [low, high) and whose phases are uniformly sampled in [-π, π].
        
        Parameters:
            size:
                Shape of the output tensor.
            low:
                Lower bound for amplitude; must be greater than or equal to 0.
            high:
                Upper bound for amplitude.
            device:
                Device of the output tensor. If None, the tensor is created on the generator's device.
        
        Returns:
            torch.Tensor: Complex64 tensor of shape `size` with sampled complex values.
        
        Raises:
            ValueError: If `low < 0`.
        """
        if low < 0:
            raise ValueError('low/high refer to the amplitude and must be positive')
        amp = self.float32_tensor(size, low, high, device=device)
        phase = self.float32_tensor(size, -torch.pi, torch.pi, device=device)
        return (amp * torch.exp(1j * phase)).to(dtype=torch.complex64)

    def complex128_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: float = 0.0,
        high: float = 1.0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate a complex128 tensor with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.
        device
            Device of the output tensor. If `None`, the tensor is created on the generator device.

        Returns
        -------
            Tensor of complex128 random numbers.
        """
        if low < 0:
            raise ValueError('low/high refer to the amplitude and must be positive')
        amp = self.float64_tensor(size, low, high, device=device)
        phase = self.float64_tensor(size, -torch.pi, torch.pi, device=device)
        return (amp * torch.exp(1j * phase)).to(dtype=torch.complex128)

    def int8_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: int = -1 << 7,
        high: int = 1 << 7,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate an int8 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).
        device
            Device of the output tensor. If `None`, the tensor is created on the generator device.

        Returns
        -------
            Tensor of int8 random numbers.
        """
        return self._randint(size, low, high, dtype=torch.int8, device=device)

    def int16_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: int = -1 << 15,
        high: int = 1 << 15,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generate an int16 tensor of uniformly distributed random values in [low, high).
        
        Parameters:
            size (Sequence[int] | int): Shape of the output tensor.
            low (int): Lower bound (inclusive).
            high (int): Upper bound (exclusive).
            device (torch.device | str | None): Device for the output tensor; if `None`, the tensor is produced on the generator's device.
        
        Returns:
            torch.Tensor: Tensor of dtype `torch.int16` with the requested shape containing random integers in [low, high).
        """
        return self._randint(size, low, high, dtype=torch.int16, device=device)

    def int32_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: int = -1 << 31,
        high: int = 1 << 31,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate an int32 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).
        device
            Device of the output tensor. If `None`, the tensor is created on the generator device.

        Returns
        -------
            Tensor of int32 random numbers.
        """
        return self._randint(size, low, high, dtype=torch.int32, device=device)

    def int64_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: int = -1 << 63,
        high: int = (1 << 63) - 1,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generate an int64 tensor whose entries are uniformly sampled from [low, high).
        
        Parameters:
            size:
                Shape of the output tensor (an int or a sequence of ints).
            low:
                Lower bound (inclusive).
            high:
                Upper bound (exclusive). Maximum allowed value is (1 << 63) - 1 due to PyTorch int64 limits.
            device:
                Device for the output tensor. If None, the tensor is created on the generator's device.
        
        Returns:
            Tensor of dtype `torch.int64` with the requested shape, containing values in [low, high).
        """
        return self._randint(size, low, high, dtype=torch.int64, device=device)

    def uint8_tensor(
        self,
        size: Sequence[int] | int = (1,),
        low: int = 0,
        high: int = 1 << 8,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generates a tensor of random `torch.uint8` values sampled uniformly from [low, high).
        
        Parameters:
            size (Sequence[int] | int): Shape of the output tensor.
            low (int): Inclusive lower bound of the sampled values.
            high (int): Exclusive upper bound of the sampled values.
            device (torch.device | str | None): Device for the output tensor; if `None`, uses the generator's device.
        
        Returns:
            torch.Tensor: Tensor with dtype `torch.uint8` containing values in the interval [low, high).
        """
        return self._randint(size, low, high, dtype=torch.uint8, device=device)

    def bool_tensor(self, size: Sequence[int] | int = (1,), device: torch.device | str | None = None) -> torch.Tensor:
        """
        Generate a boolean tensor with uniformly random True/False values.
        
        Parameters:
            size (Sequence[int] | int): Shape of the output tensor. Defaults to (1,).
            device (torch.device | str | None): Device for the output tensor. If None, the tensor is created on the generator's device.
        
        Returns:
            torch.Tensor: Tensor of dtype `torch.bool` containing randomly sampled `True` or `False` values.
        """
        return self.uint8_tensor(size, low=0, high=2, device=device).bool()

    def bool(self) -> bool:
        """
        Produce a random boolean value.
        
        Returns:
            `true` if the sampled value equals 1, `false` otherwise.
        """
        return self.uint8(0, 2) == 1

    def float32(self, low: float = 0.0, high: float = 1.0) -> float:
        """
        Generate a uniformly distributed scalar in [low, high).
        
        Parameters:
            low (float): Lower bound (inclusive).
            high (float): Upper bound (exclusive).
        
        Returns:
            float: A random `float32` value sampled uniformly from [low, high).
        """
        return self.float32_tensor((1,), low, high).item()

    def float64(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate a float64 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Random float64 number.
        """
        return self.float64_tensor((1,), low, high).item()

    def complex64(self, low: float = 0, high: float = 1.0) -> complex:
        """Generate a complex64 scalar with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Random complex64 number.
        """
        return self.complex64_tensor((1,), low, high).item()

    def complex128(self, low: float = 0, high: float = 1.0) -> complex:
        """Generate a complex128 scalar with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Random complex128 number.
        """
        return self.complex128_tensor((1,), low, high).item()

    def uint8(self, low: int = 0, high: int = (1 << 8) - 1) -> int:
        """Generate a uint8 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint8 number.
        """
        return int(self.uint8_tensor((1,), low, high).item())

    def uint16(self, low: int = 0, high: int = 1 << 16) -> int:
        """Generate a uint16 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint16 number.
        """
        if low < 0 or high > 1 << 16:
            raise ValueError('Low must be positive and high must be <= 2^16')
        return int(self.int32_tensor((1,), low, high).item())

    def uint32(self, low: int = 0, high: int = 1 << 32) -> int:
        """Generate a uint32 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint32 number.
        """
        if low < 0 or high > 1 << 32:
            raise ValueError('Low must be positive and high must be <= 2^32')
        return int(self.int64_tensor((1,), low, high).item())

    def int8(self, low: int = -1 << 7, high: int = 1 << 7) -> int:
        """Generate an int8 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int8 number.
        """
        return int(self.int8_tensor((1,), low, high).item())

    def int16(self, low: int = -1 << 15, high: int = 1 << 15) -> int:
        """Generate an int16 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int16 number.
        """
        return int(self.int16_tensor((1,), low, high).item())

    def int32(self, low: int = -1 << 31, high: int = 1 << 31) -> int:
        """Generate an int32 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int32 number.
        """
        return int(self.int32_tensor((1,), low, high).item())

    def int64(self, low: int = -1 << 63, high: int = (1 << 63) - 1) -> int:
        """Generate an int64 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int64 number.
        """
        return int(self.int64_tensor((1,), low, high).item())

    def uint64(self, low: int = 0, high: int = (1 << 64) - 1) -> int:
        """Generate a uint64 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint64 number.
        """
        if low < 0 or high > 1 << 64:
            raise ValueError('Low must be positive and high must be <= 2^64')
        range_ = high - low
        new_low = -1 << 63
        new_high = new_low + range_
        value = self.int64(new_low, new_high) - new_low + low
        return value

    def float32_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[float, ...]:
        """Generate a tuple of float32 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Tuple of float32 random numbers.
        """
        return tuple(self.float32_tensor((size,), low, high))

    def float64_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[float, ...]:
        """Generate a tuple of float64 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Tuple of float64 random numbers.
        """
        return tuple(self.float64_tensor((size,), low, high))

    def complex64_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[complex, ...]:
        """Generate a tuple of complex64 numbers with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Tuple of complex64 random numbers.
        """
        return tuple(self.complex64_tensor((size,), low, high))

    def complex128_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[complex, ...]:
        """Generate a tuple of complex128 numbers with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Tuple of complex128 random numbers.
        """
        return tuple(self.complex128_tensor((size,), low, high))

    def uint8_tuple(self, size: int, low: int = 0, high: int = 1 << 8) -> tuple[int, ...]:
        """Generate a tuple of uint8 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint8 random numbers.
        """
        return tuple(self.uint8_tensor((size,), low, high))

    def uint16_tuple(self, size: int, low: int = 0, high: int = 1 << 16) -> tuple[int, ...]:
        """Generate a tuple of uint16 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint16 random numbers.
        """
        return tuple([self.uint16(low, high) for _ in range(size)])

    def uint32_tuple(self, size: int, low: int = 0, high: int = 1 << 32) -> tuple[int, ...]:
        """Generate a tuple of uint32 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint32 random numbers.
        """
        return tuple([self.uint32(low, high) for _ in range(size)])

    def uint64_tuple(self, size: int, low: int = 0, high: int = (1 << 64) - 1) -> tuple[int, ...]:
        """Generate a tuple of uint64 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint64 random numbers.
        """
        return tuple([self.uint64(low, high) for _ in range(size)])

    def int8_tuple(self, size: int, low: int = -1 << 7, high: int = 1 << 7) -> tuple[int, ...]:
        """Generate a tuple of int8 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int8 random numbers.
        """
        return tuple(self.int8_tensor((size,), low, high))

    def int16_tuple(self, size: int, low: int = -1 << 15, high: int = 1 << 15) -> tuple[int, ...]:
        """Generate a tuple of int16 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int16 random numbers.
        """
        return tuple(self.int16_tensor((size,), low, high))

    def int32_tuple(self, size: int, low: int = -1 << 31, high: int = 1 << 31) -> tuple[int, ...]:
        """Generate a tuple of int32 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int32 random numbers.
        """
        return tuple(self.int32_tensor((size,), low, high))

    def int64_tuple(self, size: int, low: int = -1 << 63, high: int = (1 << 63) - 1) -> tuple[int, ...]:
        """Generate a tuple of int64 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int64 random numbers.
        """
        return tuple(self.int64_tensor((size,), low, high))

    def ascii(self, size: int) -> str:
        """Generate a random ASCII string.

        Parameters
        ----------
        size
            Length of the string.

        Returns
        -------
            Random ASCII string.
        """
        return ''.join([chr(self.uint8(32, 127)) for _ in range(size)])

    def rand_like(self, x: torch.Tensor, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """
        Create a tensor matching x's shape and device filled with uniform random values in [low, high).
        
        Parameters:
            x (torch.Tensor): Reference tensor whose shape, dtype, and device are used.
            low (float): Lower bound (inclusive) of the uniform interval.
            high (float): Upper bound (exclusive) of the uniform interval.
        
        Returns:
            torch.Tensor: A tensor with the same shape, dtype, and device as `x` containing values sampled uniformly from [low, high).
        """
        return self.rand_tensor(x.shape, x.dtype, low=low, high=high, device=x.device)

    def randn_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a tensor matching `x`'s shape, dtype, and device populated with standard normal (mean 0, std 1) samples.
        
        Parameters:
            x (torch.Tensor): Reference tensor whose shape, dtype, and device are used.
        
        Returns:
            torch.Tensor: Tensor with the same shape, dtype, and device as `x` containing samples from a standard normal distribution.
        """
        return self.randn_tensor(x.shape, x.dtype, device=x.device)

    def rand_tensor(
        self,
        size: Sequence[int],
        dtype: torch.dtype,
        low: float | int = 0,
        high: int | float = 1,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Generate a tensor of the given shape and dtype with values sampled from a uniform distribution over the specified range.
        
        Parameters:
            size (Sequence[int]): Shape of the output tensor.
            dtype (torch.dtype): Desired data type of the output tensor.
            low (float | int): Lower bound of the sampling interval (inclusive for floats/complex amplitude).
            high (float | int): Upper bound of the sampling interval (exclusive for floats/complex amplitude).
            device (torch.device | str | None): Device to create the tensor on; if None, uses the generator's/default device.
        
        Returns:
            torch.Tensor: Tensor of shape `size` and dtype `dtype` containing random samples:
                - For floating dtypes: values in [low, high).
                - For complex dtypes: complex numbers with amplitude in [low, high) and phase in [-π, π].
                - For bool: values equally likely `True` or `False`.
                - For integer dtypes: integer values in [ceil(low), floor(high)).
        """
        if dtype.is_complex:
            real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            amp = self._rand(size, low, high, real_dtype, device=device)
            phase = self._rand(size, -torch.pi, torch.pi, real_dtype, device=device)
            tensor = (amp * torch.exp(1j * phase)).to(dtype=dtype)
        elif dtype.is_floating_point:
            tensor = self._rand(size, low, high, dtype, device=device)
        elif dtype == torch.bool:
            tensor = self._randint(size, 0, 2, dtype=torch.int32, device=device) > 0
        else:
            tensor = self._randint(size, ceil(low), floor(high), dtype, device=device)
        return tensor

    def randn_tensor(
        self, size: Sequence[int], dtype: torch.dtype, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """
        Generates a tensor of the requested shape and dtype with samples drawn from the standard normal distribution.
        
        Parameters:
            size (Sequence[int]): Shape of the output tensor.
            dtype (torch.dtype): Data type of the output tensor.
            device (torch.device | str | None): Device for the returned tensor. If `None`, the tensor is created on the generator's device.
        
        Returns:
            torch.Tensor: Tensor of shape `size` and dtype `dtype` containing independent samples from N(0, 1).
        """
        tensor = torch.randn(size=size, generator=self.generator, dtype=dtype, device=self.generator.device)
        return tensor.to(device=device) if device is not None else tensor

    def beta_tensor(
        self,
        size: Sequence[int] | int,
        alpha: float,
        beta: float,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Sample values from a Beta(alpha, beta) distribution with the requested shape and dtype.
        
        Parameters:
            size (int | Sequence[int]): Output shape of the returned tensor.
            alpha (float): Alpha (concentration) parameter of the Beta distribution; must be > 0.
            beta (float): Beta (concentration) parameter of the Beta distribution; must be > 0.
            dtype (torch.dtype): Data type of the returned tensor (default: torch.float32).
            device (torch.device | str | None): Optional device for the returned tensor; when provided the result is moved to this device.
        
        Returns:
            torch.Tensor: Tensor of shape `size` with values in the interval (0, 1), drawn from Beta(alpha, beta) and of the requested dtype.
        """
        size_ = (size,) if isinstance(size, int) else tuple(size)
        alpha_tensor = torch.full(size_, alpha, dtype=dtype, device=self.generator.device)
        beta_tensor = torch.full(size_, beta, dtype=dtype, device=self.generator.device)
        sample_alpha = torch._standard_gamma(alpha_tensor, generator=self.generator)
        sample_beta = torch._standard_gamma(beta_tensor, generator=self.generator)
        tensor = sample_alpha / (sample_alpha + sample_beta)
        return tensor.to(device=device) if device is not None else tensor

    def randperm(
        self, n: int, *, dtype: torch.dtype = torch.int64, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """
        Return a tensor with a random permutation of integers from 0 to n - 1.
        
        Parameters
        ----------
        n
            Number of elements to permute.
        dtype
            Data type of the output tensor.
        device
            Device for the returned tensor; if `None`, the tensor is created on the generator's device.
        
        Returns
        -------
        torch.Tensor
            A 1-D tensor of shape (n,) containing a random permutation of the integers 0..n-1 with the requested dtype.
        """
        tensor = torch.randperm(n, generator=self.generator, dtype=dtype, device=self.generator.device)
        return tensor.to(device=device) if device is not None else tensor

    def gaussian_variable_density_samples(
        self, shape: Sequence[int], low: int, high: int, fwhm: float = float('inf'), always_sample: Sequence[int] = ()
    ) -> torch.Tensor:
        """
        Generate a set of integer indices sampled from [low, high) with a Gaussian-shaped sampling density.
        
        Parameters:
            shape (Sequence[int]): Output shape interpreted as (*batch_dims, n_samples). Sampling is performed along the last dimension; all preceding dimensions are treated as independent batches.
            low (int): Inclusive lower bound of the integer domain.
            high (int): Exclusive upper bound of the integer domain.
            fwhm (float): Full-width at half-maximum of the Gaussian weight used to bias sampling (larger values → flatter distribution). Defaults to infinity (uniform sampling).
            always_sample (Sequence[int]): Sequence of indices that must appear in every sample (each value should be in [low, high)).
        
        Returns:
            torch.Tensor: Integer tensor of shape (*batch_dims, n_samples) containing sorted indices in [low, high). Each batch contains the requested number of samples and includes the indices from `always_sample`.
        
        Raises:
            ValueError: If n_samples (last element of `shape`) is greater than (high - low), or if more always-sampled indices are requested than n_samples.
        """
        *n_batch, n_samples = shape
        if n_samples > high - low:
            raise ValueError('n_samples must be <= (high - low)')
        n_random = n_samples - len(always_sample)
        if n_random < 0:
            raise ValueError('more always sampled points requested than total number of samples')
        elif n_random == 0:
            return torch.tensor(always_sample).sort().values.broadcast_to(*n_batch, -1)
        pdf = torch.exp(-torch.tensor(2.0).log() * (2 * torch.arange(low, high) / fwhm) ** 2)
        pdf[[x - low for x in always_sample]] = 0
        if len(shape) > 1:
            pdf = pdf.broadcast_to((*n_batch, -1)).flatten(end_dim=-2)

        idx_rand = pdf.multinomial(n_random, False, generator=self.generator) + low
        if len(shape) > 1:
            idx_rand = idx_rand.unflatten(0, n_batch)
        idx_always = torch.tensor(always_sample).broadcast_to(*n_batch, -1)
        return torch.cat([idx_rand, idx_always], -1).sort().values
