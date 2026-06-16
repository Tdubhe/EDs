# U1-boson-fermion

`U1-boson-fermion` is a small Python library extracted from `bf_scar.ipynb`. It provides helpers for:

- converting QLM bitstrings to lattice arrays and back
- checking and applying plaquette and hopping moves
- generating reachable Krylov bases
- constructing sparse Hamiltonians
- plotting lattice charge and fermion configurations

## Install

From this folder:

```powershell
pip install -e .
```

## Quick Start

```python
import numpy as np
from U1_boson_fermion import converter_f2s, basis_generator, ham

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
