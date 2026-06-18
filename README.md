# spectral-lgt

`spectral-lgt` is a small Python library for spectral lattice gauge theory workflows. It provides helpers for:

- converting QLM bitstrings to lattice arrays and back
- checking and applying plaquette and hopping moves
- generating reachable Krylov bases
- constructing sparse Hamiltonians
- plotting lattice charge and fermion configurations
- exact diagonalization and real-time dynamics

Model-specific code lives under `spectral_lgt.models`. The current model is:

- `spectral_lgt.models.u1`: U(1) quantum link model basis, converters, Hamiltonians, and plotting

General spectral tools live at the package level:

- `spectral_lgt.simulation`: exact diagonalization and time evolution

## Install

From this folder:

```powershell
pip install -e .
```

## Quick Start

```python
import numpy as np
from spectral_lgt import converter_f2s, basis_generator, ham

Lx = Ly = 4
num = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
])
hor = np.ones((Ly, Lx))
ver = np.ones((Ly, Lx))

initial = converter_f2s("file2binary", [num, hor, ver], s=1)
basis, basis_names = basis_generator(
    initial,
    Lx=Lx,
    Ly=Ly,
    m=0,
    g=0,
    ev=np.inf,
    dynamics="both",
    S=1,
)

H = ham(
    basis,
    basis_names,
    Lx=Lx,
    Ly=Ly,
    m=0.75,
    J=0,
    t=10,
    g=0,
    ev=np.inf,
    fermion=True,
    S=1,
)
```
