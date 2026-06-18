"""Exact diagonalization and time-evolution helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.sparse import issparse, spmatrix

MatrixLike = Union[ArrayLike, spmatrix]
OperatorCollection = Union[Mapping[str, MatrixLike], Sequence[MatrixLike], None]


def _dense(matrix: MatrixLike) -> NDArray[Any]:
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def ed(
    H: MatrixLike,
    backend: str = "cpu",
    return_eigenvectors: bool = True,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Diagonalize a Hermitian Hamiltonian on CPU or GPU.

    GPU mode requires CuPy. Results are returned as NumPy arrays.
    """
    backend = backend.lower()
    if backend == "cpu":
        vals, vecs = np.linalg.eigh(np.asarray(_dense(H), dtype=np.complex128))
        return (vals, vecs) if return_eigenvectors else vals

    if backend == "gpu":
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError("backend='gpu' requires cupy.") from exc

        vals, vecs = cp.linalg.eigh(cp.asarray(_dense(H), dtype=cp.complex128))
        vals = cp.asnumpy(vals)
        return (vals, cp.asnumpy(vecs)) if return_eigenvectors else vals

    raise ValueError("backend must be 'cpu' or 'gpu'.")


def _operator_items(operators: OperatorCollection) -> Iterable[tuple[str | int, MatrixLike]]:
    if operators is None:
        return []
    if isinstance(operators, dict):
        return operators.items()
    return enumerate(operators)


def _expectation(psi: NDArray[np.complex128], operator: MatrixLike) -> complex:
    op = np.asarray(_dense(operator), dtype=np.complex128)
    return np.vdot(psi, op @ psi)


def _spectrum_dynamics(
    H: MatrixLike,
    psi0: NDArray[np.complex128],
    times: NDArray[np.float64],
    backend: str,
) -> NDArray[np.complex128]:
    vals, vecs = ed(H, backend=backend, return_eigenvectors=True)
    coeffs = vecs.conj().T @ psi0
    return np.asarray([vecs @ (np.exp(-1j * vals * time) * coeffs) for time in times])


def _expm_dynamics(H: MatrixLike, psi0: NDArray[np.complex128], times: NDArray[np.float64]) -> NDArray[np.complex128]:
    H = np.asarray(_dense(H), dtype=np.complex128)
    return np.asarray([expm(-1j * H * time) @ psi0 for time in times])


def simulate_dynamics(
    H: MatrixLike,
    psi0: ArrayLike,
    times: ArrayLike,
    operators: OperatorCollection = None,
    method: str = "spectrum",
    backend: str = "cpu",
    return_psi_t: bool = False,
) -> dict[str, Any]:
    """Evolve ``psi0`` and measure a set of operators.

    ``method="spectrum"`` uses :func:`ed` and supports ``backend="cpu"`` or
    ``backend="gpu"``. ``method="expm"`` uses SciPy on CPU. ``operators`` may
    be either a list/tuple or a dict of named operators. The result contains all
    expectation values and either ``psi_t`` or only ``psi_final``.
    """
    method = method.lower()
    backend = backend.lower()
    times = np.asarray(times, dtype=float)
    psi0 = np.asarray(psi0, dtype=np.complex128)

    if method == "spectrum":
        psi_t = _spectrum_dynamics(H, psi0, times, backend)
    elif method == "expm":
        if backend != "cpu":
            raise ValueError("method='expm' only supports backend='cpu'.")
        psi_t = _expm_dynamics(H, psi0, times)
    else:
        raise ValueError("method must be 'spectrum' or 'expm'.")

    expectations = {
        name: np.asarray([_expectation(psi, op) for psi in psi_t])
        for name, op in _operator_items(operators)
    }
    result = {"times": times, "expectations": expectations}
    result["psi_t" if return_psi_t else "psi_final"] = psi_t if return_psi_t else psi_t[-1]
    return result
