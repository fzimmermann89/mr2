"""ISMRMRD parsing and transport helpers, independent of KData normalization."""

from collections.abc import Generator
from contextlib import nullcontext
from os import PathLike
from pathlib import Path
from typing import BinaryIO, cast

import ismrmrd
import ismrmrd.serialization


class _ReadExact:
    """Adapt blocking binary inputs with short reads to the ISMRMRD deserializer."""

    def __init__(self, stream: BinaryIO) -> None:
        """Wrap a caller-owned binary stream."""
        self.stream = stream

    def read(self, size: int) -> bytes:
        """Read the requested bytes, or fewer only at EOF."""
        chunks = []
        remaining = size
        while remaining:
            chunk = self.stream.read(remaining)
            if chunk is None:
                raise BlockingIOError('ISMRMRD input must be a blocking binary stream.')
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)


def read_stream(
    source: str | PathLike | BinaryIO,
) -> Generator[tuple[ismrmrd.xsd.ismrmrdHeader, list[ismrmrd.Acquisition]], None, None]:
    """Open a protocol source and group measurements without owning caller streams."""
    with (
        Path(source).open('rb') if isinstance(source, (str, PathLike)) else nullcontext(source) as stream,
        ismrmrd.serialization.ProtocolDeserializer(cast(BinaryIO, _ReadExact(stream))) as reader,
    ):
        header = None
        acquisitions: list[ismrmrd.Acquisition] = []
        for obj in reader.deserialize():
            if isinstance(obj, ismrmrd.xsd.ismrmrdHeader):
                if header is not None:
                    yield header, acquisitions
                header, acquisitions = obj, []
            elif isinstance(obj, ismrmrd.Acquisition):
                if header is None:
                    raise ValueError('ISMRMRD stream contains acquisitions before the first header.')
                acquisitions.append(obj)
                del obj
        if header is not None:
            yield header, acquisitions
