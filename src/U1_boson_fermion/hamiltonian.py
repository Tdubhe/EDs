"""Sparse Hamiltonian construction."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix

from .converters import converter_b2f
from .dynamics import (
    apply_hopping_spinS_fast,
    apply_plaquette_spinS_fast,
    hopping_checker,
    plaquette_checker,
)


def ham(basis, basis_name, Lx, Ly, m, J, t, g, ev=0.1, fermion=False, S=1 / 2, pbc=False):
    """Construct the sparse Hamiltonian for a generated basis."""

    def compute_energy(bi, lx, ly, m, g):
        num, ver, hor = converter_b2f(bi, lx, ly, S)
        onsite_energy = m * sum(num[y, x] * (-1) ** (x + y) for x in range(lx) for y in range(ly))
        if S == 1 / 2:
            link_energy = g * (np.sum(hor) + np.sum(ver))
        else:
            link_energy = g**2 * (np.sum(hor**2) + np.sum(ver**2)) / 2
        return onsite_energy + link_energy

    if S != 1 / 2 and g != 0:
        J = 1 / (g**2)

    def count_particles_between(bitstring, i, j):
        a, b = sorted((i, j))
        n_particles = 0
        for k in range(a + 1, b):
            if k % 3 == 0:
                n_particles += int(bitstring[k])
        return n_particles

    dim = len(basis_name)
    H = lil_matrix((dim, dim), dtype=np.float64)

    for i in range(dim):
        state = basis_name[i]
        H[i, i] = compute_energy(state, Lx, Ly, m, g)

        for ind, indices, direction in hopping_checker(state, basis_name[0], Lx, Ly, S, ev, m, g, pbc):
            x = ind // Ly
            staggered_sign = (-1) ** x
            fermion_sign = 1

            if fermion:
                i_bit = indices[0]
                if direction in ["hop_up", "hop_down"]:
                    j_bit = indices[3]
                elif direction in ["hop_left", "hop_right"]:
                    j_bit = indices[4]
                else:
                    raise ValueError(f"Unknown hopping direction: {direction}")

                fermion_sign = (-1) ** count_particles_between(state, i_bit, j_bit)

            new_state = apply_hopping_spinS_fast(state, indices, direction, S)
            if new_state not in basis:
                continue
            j = basis[new_state][0]

            if direction in ["hop_up", "hop_down"]:
                H[i, j] += -staggered_sign * t * fermion_sign / 2
            else:
                H[i, j] += -t * fermion_sign / 2

        for _, indices, direction in plaquette_checker(state, Lx, Ly, S, conserveenergy=False, pbc=pbc):
            new_state = apply_plaquette_spinS_fast(state, indices, direction)
            if new_state not in basis:
                continue
            j = basis[new_state][0]
            H[i, j] += -J

    return H.tocsr()
