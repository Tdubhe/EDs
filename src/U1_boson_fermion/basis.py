"""Basis construction helpers."""

from __future__ import annotations

from .dynamics import generate_all_reachable_configs


def basis_generator(initial_state, Lx, Ly, m, g, ev, dynamics="both", S=1 / 2, pbc=False, progress=False):
    """Generate a basis dictionary and ordered state-name list."""
    basis = {}
    name_list = []

    states = list(
        generate_all_reachable_configs(
            initial_state=initial_state,
            Lx=Lx,
            Ly=Ly,
            S=S,
            conserveenergy=True,
            dynamics=dynamics,
            m=m,
            g=g,
            ev=ev,
            pbc=pbc,
            progress=progress,
        )
    )
    ordered_states = sorted(states, key=lambda x: x[1])

    for state, order in ordered_states:
        basis[state] = [len(basis), order]
        name_list.append(state)

    return basis, name_list
