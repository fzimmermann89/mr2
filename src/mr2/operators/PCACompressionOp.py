"""PCA Compression Operator."""

import einops
import torch
from einops import repeat

from mr2.algorithms.varimax import varimax
from mr2.operators.LinearOperator import LinearOperator


class PCACompressionOp(LinearOperator):
    """PCA based compression operator."""

    def __init__(
        self,
        data: torch.Tensor,
        n_components: int,
        centering: bool = True,
        rotate: bool = False,
    ) -> None:
        """Construct a PCA based compression operator.

        The operator carries out an SVD of the correlation followed by a threshold of the `n_components` largest values
        along the last dimension of a data with shape `(*other, joint_dim, compression_dim)`.
        A single SVD is carried out for everything along `joint_dim`. `other` are batch dimensions.

        You should disable centering for MRF subspace reconstruction, and keep it enabled for coil compression.

        Consider combining this operator with `~mr2.operators.RearrangeOp` to make sure the data is
        in the correct shape before applying.

        Parameters
        ----------
        data
            Data of shape `(*other, joint_dim, compression_dim)` to be used to find the principal components.
        n_components
            Number of principal components to keep along the compression_dim.
        centering
            Should the data be centered? With centering, only fluctuations around the mean are encoded in the
            subspace. You should not use centering for qMRI signal compression.
        rotate
            Apply a varimax rotation to the compression matrix to distribute the signal more equally across the
            subspace.
        """
        super().__init__()
        if centering:
            data = data - data.mean(-2, keepdim=True)
        # different compression matrices along the *other dimensions
        correlation = einops.einsum(data.conj(), data, '... joint comp1, ... joint comp2 -> ... comp1 comp2')
        _eigenvalues, v = torch.linalg.eigh(correlation)  # faster then svd if we only care about V
        # Forward applies V^T to column samples, equivalent to data @ V for voxel rows.
        # Add joint_dim along which the compression is the same.
        v = repeat(v, '... comp1 comp2 -> ... joint_dim  comp2 comp1', joint_dim=1)
        v = v[..., -n_components:, :].flip(-2)  # V is sorted in ascending order
        if rotate:
            v = varimax(v)
        self._compression_matrix = v

    def __call__(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply PCA-based compression to the input data.

        The data is projected onto the principal components determined during
        the operator's initialization.

        Parameters
        ----------
        data
            Input data to be compressed. Expected shape is
            `(*other, joint_dim, compression_dim)`.

        Returns
        -------
            Compressed data, with shape `(*other, joint_dim, n_components)`.
        """
        return super().__call__(data)

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of PCACompressionOp.

        .. note::
            Prefer calling the instance of the PCACompressionOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        try:
            self._check_batch_shape(data)
            result = (self._compression_matrix @ data.unsqueeze(-1)).squeeze(-1)
        except RuntimeError as e:
            raise RuntimeError(
                'Shape or device mismatch in Compression: '
                f'Matrix {tuple(self._compression_matrix.shape)} on {self._compression_matrix.device} '
                f'cannot be multiplied with Data {tuple(data.shape)} on {data.device}.'
            ) from e
        return (result,)

    def adjoint(self, data: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of PCA-based compression (expansion).

        The data, assumed to be in the compressed principal component space,
        is projected back to the original data space using the hermitian
        transpose of the compression matrix.

        Parameters
        ----------
        data
            Compressed input data. Expected shape is
            `(*other, joint_dim, n_components)`.

        Returns
        -------
            Expanded data, with shape `(*other, joint_dim, compression_dim)`.
        """
        try:
            self._check_batch_shape(data)
            result = (self._compression_matrix.mH @ data.unsqueeze(-1)).squeeze(-1)
        except RuntimeError as e:
            raise RuntimeError(
                'Shape or device mismatch in adjoint Compression: '
                f'Matrix^H {tuple(self._compression_matrix.mH.shape)} on {self._compression_matrix.device} '
                f'cannot be multiplied with Data {tuple(data.shape)} on {data.device}.'
            ) from e
        return (result,)

    def _check_batch_shape(self, data: torch.Tensor) -> None:
        """Reject batch expansion before allocating the matrix product."""
        matrix_batch = self._compression_matrix.shape[:-2]
        data_batch = data.shape[:-1]
        offset = len(data_batch) - len(matrix_batch)
        if offset >= 0 and all(
            size == 1 or size == data_batch[offset + axis] for axis, size in enumerate(matrix_batch)
        ):
            return
        # Let incompatible shapes raise RuntimeError through the existing error handler.
        if torch.broadcast_shapes(matrix_batch, data_batch) != data_batch:
            raise ValueError('PCA input must have fully expanded batch dimensions.')

    @property
    def compression_matrix(self) -> torch.Tensor:
        """Get the compression matrix."""
        return self._compression_matrix

    def _pinv(self, *, tolerance: float, max_iterations: int) -> LinearOperator:  # noqa: ARG002
        """Return the exact pseudoinverse on the declared tensor spaces."""
        return self.H
