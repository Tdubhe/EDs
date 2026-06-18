"""Sparse Hamiltonian construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .converters import converter_b2f
from .dynamics import (
    apply_hopping_spinS_fast,
    apply_plaquette_spinS_fast,
    hopping_checker,
    plaquette_checker,
)

BasisMap = dict[str, list[int]]


def ham(
    basis: BasisMap,
    basis_name: Sequence[str],
    Lx: int,
    Ly: int,
    m: float,
    J: float,
    t: float,
    g: float,
    ev: float = 0.1,
    fermion: bool = False,
    S: float = 1 / 2,
    pbc: bool = False,
    auto_J_from_g: bool = False,
) -> csr_matrix:
    """Construct the sparse Hamiltonian for a generated basis."""

    def compute_energy(bi: str, lx: int, ly: int, m: float, g: float) -> float:
        num, ver, hor = converter_b2f(bi, lx, ly, S)
        onsite_energy = m * sum(num[y, x] * (-1) ** (x + y) for x in range(lx) for y in range(ly))
        if S == 1 / 2:
            link_energy = g * (np.sum(hor) + np.sum(ver))
        else:
            link_energy = g**2 * (np.sum(hor**2) + np.sum(ver**2)) / 2
        return onsite_energy + link_energy

    if auto_J_from_g and S != 1 / 2 and g != 0:
        J = 1 / (g**2)

    def count_particles_between(bitstring: str, i: int, j: int) -> int:
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


def trotter_ham(
    basis: BasisMap,
    basis_name: Sequence[str],
    Lx: int,
    Ly: int,
    m: float,
    J: float,
    t: float,
    g: float,
    ev: float = np.inf,
    fermion: bool = False,
    S: float = 1 / 2,
    pbc: bool = False,
    auto_J_from_g: bool = False,
) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    def compute_energy(bi: str, lx: int, ly: int, m: float, g: float) -> float:
        num, ver, hor = converter_b2f(bi, lx, ly, S)
        onsite_energy = m * sum(num[y, x] * (-1) ** (x + y) for x in range(lx) for y in range(ly))
        if S == 1 / 2:
            link_energy = g * (np.sum(hor) + np.sum(ver))
        else:
            link_energy = g**2 * (np.sum(hor**2) + np.sum(ver**2)) / 2
        return onsite_energy + link_energy
     

    def count_particles_between(bitstring: str, i: int, j: int) -> int:
        a, b = sorted((i, j))
        n_particles = 0
        for k in range(a + 1, b):
            if k % 3 == 0:
                n_particles += int(bitstring[k])
        return n_particles

    if auto_J_from_g and S != 1 / 2 and g != 0:
        J = 1 / (g**2)


    dim = len(basis_name)
    H1 = lil_matrix((dim, dim), dtype=np.float64)
    H2 = lil_matrix((dim, dim), dtype=np.float64)
    H3 = lil_matrix((dim, dim), dtype=np.float64)   
    H4 = lil_matrix((dim, dim), dtype=np.float64)
    H5 = lil_matrix((dim, dim), dtype=np.float64)

    for i in range(dim):
        state = basis_name[i]
        H5[i, i] = compute_energy(state, Lx, Ly, m, g)

        for ind, indices, direction in hopping_checker(state, basis_name[0], Lx, Ly, S, ev, m, g, pbc):
            x = ind // Ly
            y = ind % Ly
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

            if direction in ["hop_left", "hop_right"] and x%2 == 0:
                H1[i, j] += -t * fermion_sign / 2
            
            if direction in ["hop_left", "hop_right"] and x%2 == 1:
                H2[i, j] += -t * fermion_sign / 2
            
            if direction in ["hop_up", "hop_down"] and y%2 == 0:
                H3[i, j] += -staggered_sign * t * fermion_sign / 2
            if direction in ["hop_up", "hop_down"] and y%2 == 1:
                H4[i, j] += -staggered_sign * t * fermion_sign / 2

        for _, indices, direction in plaquette_checker(state, Lx, Ly, S, conserveenergy=False, pbc=pbc):
            new_state = apply_plaquette_spinS_fast(state, indices, direction)
            if new_state not in basis:
                continue
            j = basis[new_state][0]
            H5[i, j] += -J

    return H1.tocsr(), H2.tocsr(), H3.tocsr(), H4.tocsr(), H5.tocsr()



def mag(
    basis: BasisMap,
    basis_name: Sequence[str],
    Lx: int,
    Ly: int,
    J: float,
    S: float = 1 / 2,
    ind_Set: set[int] | Sequence[int] | None = None,
    pbc: bool = False,
) -> csr_matrix:
    """Construct the plaquette magnetization x-operator from plaquette flips."""
    dim = len(basis_name)

    H = lil_matrix((dim, dim), dtype=np.complex128)

    for i in range(dim):
        state = basis_name[i]

        pla_l = plaquette_checker(
            state, Lx, Ly, S, conserveenergy=False, pbc=pbc
        )
        for ind, indices, direction in pla_l:
            # ind_Set restricts the operator to selected plaquette positions.
            if ind_Set is None or ind in ind_Set:
                new_state = apply_plaquette_spinS_fast(
                    state, indices, direction
                )
                if new_state not in basis:
                    continue
                j = basis[new_state][0]
                H[i, j] += J

    return H.tocsr()


def mag_y(
    basis: BasisMap,
    basis_name: Sequence[str],
    Lx: int,
    Ly: int,
    J: float,
    S: float = 1 / 2,
    ind_Set: set[int] | Sequence[int] | None = None,
    pbc: bool = False,
) -> csr_matrix:
    """Construct the plaquette magnetization y-operator from plaquette flips."""
    dim = len(basis_name)

    H = lil_matrix((dim, dim), dtype=np.complex128)

    for i in range(dim):
        state = basis_name[i]

        pla_l = plaquette_checker(
            state, Lx, Ly, S, conserveenergy=False, pbc=pbc
        )

        for ind, indices, direction in pla_l:
            if ind_Set is None or ind in ind_Set:
                new_state = apply_plaquette_spinS_fast(
                    state, indices, direction
                )
                if new_state not in basis:
                    continue
                j = basis[new_state][0]

                # Forward/backward plaquette flips are the two off-diagonal
                # directions of sigma_y, so their amplitudes are complex conjugates.
                if direction == "forward":
                    H[i, j] += 1j * J
                else:
                    H[i, j] += -1j * J
    return H.tocsr()
