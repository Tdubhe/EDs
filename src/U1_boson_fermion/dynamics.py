"""Plaquette, hopping, and reachable-configuration generation."""

from __future__ import annotations

from collections import deque

import numpy as np

from .converters import converter_b2f


def plaquette_checker(bitstring: str, Lx: int, Ly: int, S: float, conserveenergy: bool, pbc: bool = False):
    """Find plaquette actions available from a state."""
    max_m = int(2 * S)
    surviving_plaquettes = []

    def site_index(x, y):
        return x * Ly + y

    def bit_indices(x, y):
        base = 3 * site_index(x, y)
        return base, base + 1, base + 2

    DX = Lx if pbc else Lx - 1
    DY = Ly if pbc else Ly - 1

    for x in range(DX):
        for y in range(DY):
            try:
                if x == Lx - 1 and y < Ly - 1:
                    _, _, A_up = bit_indices(x, y)
                    _, B_right, _ = bit_indices(x, y + 1)
                    _, A_right, _ = bit_indices(x, y)
                    _, _, C_up = bit_indices(0, y)
                elif x < Lx - 1 and y == Ly - 1:
                    _, _, A_up = bit_indices(x, y)
                    _, B_right, _ = bit_indices(x, 0)
                    _, A_right, _ = bit_indices(x, y)
                    _, _, C_up = bit_indices(x + 1, y)
                elif x == Lx - 1 and y == Ly - 1:
                    _, _, A_up = bit_indices(x, y)
                    _, B_right, _ = bit_indices(x, 0)
                    _, A_right, _ = bit_indices(x, y)
                    _, _, C_up = bit_indices(0, y)
                else:
                    _, _, A_up = bit_indices(x, y)
                    _, B_right, _ = bit_indices(x, y + 1)
                    _, A_right, _ = bit_indices(x, y)
                    _, _, C_up = bit_indices(x + 1, y)

                indices = [A_up, B_right, A_right, C_up]
                ms = [int(bitstring[idx]) for idx in indices]
                n_zero_before = sum(m == 0 for m in ms)

                forward = ms[0] < max_m and ms[1] < max_m and ms[2] > 0 and ms[3] > 0
                if forward:
                    ms_after = [ms[0] + 1, ms[1] + 1, ms[2] - 1, ms[3] - 1]
                    n_zero_after = sum(m == 0 for m in ms_after)
                    if (not conserveenergy) or (n_zero_after == n_zero_before):
                        surviving_plaquettes.append((site_index(x, y), indices, "forward"))

                backward = ms[0] > 0 and ms[1] > 0 and ms[2] < max_m and ms[3] < max_m
                if backward:
                    ms_after = [ms[0] - 1, ms[1] - 1, ms[2] + 1, ms[3] + 1]
                    n_zero_after = sum(m == 0 for m in ms_after)
                    if (not conserveenergy) or (n_zero_after == n_zero_before):
                        surviving_plaquettes.append((site_index(x, y), indices, "backward"))

            except IndexError:
                continue

    return surviving_plaquettes


def apply_plaquette_spinS_fast(state, indices, direction: str) -> str:
    """Apply a pre-validated spin-S plaquette operation."""
    new_state = list(state) if isinstance(state, str) else state.copy()

    if direction == "forward":
        new_state[indices[0]] = str(int(new_state[indices[0]]) + 1)
        new_state[indices[1]] = str(int(new_state[indices[1]]) + 1)
        new_state[indices[2]] = str(int(new_state[indices[2]]) - 1)
        new_state[indices[3]] = str(int(new_state[indices[3]]) - 1)
    elif direction == "backward":
        new_state[indices[0]] = str(int(new_state[indices[0]]) - 1)
        new_state[indices[1]] = str(int(new_state[indices[1]]) - 1)
        new_state[indices[2]] = str(int(new_state[indices[2]]) + 1)
        new_state[indices[3]] = str(int(new_state[indices[3]]) + 1)
    else:
        raise ValueError(f"Unknown plaquette direction: {direction}")

    return "".join(new_state)


def _compute_energy(bi: str, lx: int, ly: int, m: float, g: float, S: float) -> float:
    num, ver, hor = converter_b2f(bi, lx, ly, S)
    onsite_energy = m * sum(num[y, x] * (-1) ** (x + y) for x in range(lx) for y in range(ly))
    if S == 1 / 2:
        link_energy = g * (np.sum(hor) + np.sum(ver))
    else:
        link_energy = g**2 * (np.sum(hor**2) + np.sum(ver**2))
    return onsite_energy + link_energy


def hopping_checker(bitstring: str, b_int: str, Lx: int, Ly: int, S: float, ev: float, m: float, g: float, pbc: bool = False):
    """Find energy-conserving matter hopping moves available from a state."""
    surviving_hops = []
    max_m = 2 * S

    def delta_energy(ms, msa, x, y, m, g):
        s1 = (-1) ** (x + y)
        s2 = (-1) ** (x + y + 1)
        if S == 1 / 2:
            return (
                (msa[0] * s1 + msa[3] * s2 + msa[4] * s2) * m
                + msa[1] * g
                + msa[2] * g
                - (ms[0] * s1 + ms[3] * s2 + ms[4] * s2) * m
                - ms[1] * g
                - ms[2] * g
            )
        return (
            (msa[0] * s1 + msa[3] * s2 + msa[4] * s2) * m
            + (msa[1] - 1) ** 2 * g**2
            + (msa[2] - 1) ** 2 * g**2
            - (ms[0] * s1 + ms[3] * s2 + ms[4] * s2) * m
            - (ms[1] - 1) ** 2 * g**2
            - (ms[2] - 1) ** 2 * g**2
        )

    def site_index(x, y):
        return x * Ly + y

    def bit_indices(x, y):
        base = 3 * site_index(x, y)
        return base, base + 1, base + 2

    E_int = _compute_energy(b_int, Lx, Ly, m, g, S)
    E_current = _compute_energy(bitstring, Lx, Ly, m, g, S)

    for x in range(Lx):
        for y in range(Ly):
            base, r, u = bit_indices(x, y)
            if pbc:
                base_up, _, _ = bit_indices(x, y + 1) if y + 1 < Ly else bit_indices(x, 0)
                base_right, _, _ = bit_indices(x + 1, y) if x + 1 < Lx else bit_indices(0, y)
            else:
                base_up, _, _ = bit_indices(x, y + 1) if y + 1 < Ly else (np.inf, 0, 0)
                base_right, _, _ = bit_indices(x + 1, y) if x + 1 < Lx else (np.inf, 0, 0)

            indices = [base, r, u, base_up, base_right]
            ms = [int(bitstring[idx]) if idx < np.inf else 100000 for idx in indices]

            if S == 1 / 2:
                if ms[0] == 1 and ms[3] == 0 and ms[2] == 1:
                    msa = [ms[0] - 1, ms[1], (ms[2] + 1) % 2, ms[3] + 1, ms[4]]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_up"))
                if ms[0] == 0 and ms[3] == 1 and ms[2] == 0:
                    msa = [ms[0] + 1, ms[1], (ms[2] + 1) % 2, ms[3] - 1, ms[4]]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_down"))
                if ms[0] == 0 and ms[4] == 1 and ms[1] == 0:
                    msa = [ms[0] + 1, (ms[1] + 1) % 2, ms[2], ms[3], ms[4] - 1]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_left"))
                if ms[0] == 1 and ms[4] == 0 and ms[1] == 1:
                    msa = [ms[0] - 1, (ms[1] + 1) % 2, ms[2], ms[3], ms[4] + 1]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_right"))
            else:
                if ms[0] == 1 and ms[3] == 0 and ms[2] > 0:
                    msa = [ms[0] - 1, ms[1], ms[2] - 1, ms[3] + 1, ms[4]]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_up"))
                if ms[0] == 0 and ms[3] == 1 and ms[2] < max_m:
                    msa = [ms[0] + 1, ms[1], ms[2] + 1, ms[3] - 1, ms[4]]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_down"))
                if ms[0] == 0 and ms[4] == 1 and ms[1] < max_m:
                    msa = [ms[0] + 1, ms[1] + 1, ms[2], ms[3], ms[4] - 1]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_left"))
                if ms[0] == 1 and ms[4] == 0 and ms[1] > 0:
                    msa = [ms[0] - 1, ms[1] - 1, ms[2], ms[3], ms[4] + 1]
                    if abs(E_int - E_current + delta_energy(ms, msa, x, y, m, g)) < ev:
                        surviving_hops.append((site_index(x, y), indices, "hop_right"))

    return surviving_hops


def apply_hopping_spinS_fast(state, indices, direction: str, S: float) -> str:
    """Apply a pre-validated hopping operation."""
    new_state = list(state) if isinstance(state, str) else state.copy()
    m, r, u, tm, tl = indices

    if S == 1 / 2:
        if direction in ("hop_up", "hop_down"):
            new_state[m] = str(int(new_state[m]) - 1 if direction == "hop_up" else int(new_state[m]) + 1)
            new_state[u] = str((int(new_state[u]) + 1) % 2)
            new_state[tm] = str(int(new_state[tm]) + 1 if direction == "hop_up" else int(new_state[tm]) - 1)
        elif direction in ("hop_right", "hop_left"):
            new_state[m] = str(int(new_state[m]) - 1 if direction == "hop_right" else int(new_state[m]) + 1)
            new_state[r] = str((int(new_state[r]) + 1) % 2)
            new_state[tl] = str(int(new_state[tl]) + 1 if direction == "hop_right" else int(new_state[tl]) - 1)
        else:
            raise ValueError(f"Unknown hopping direction: {direction}")
        return "".join(new_state)

    if S == 1:
        if direction in ("hop_up", "hop_down"):
            new_state[m] = str(int(new_state[m]) - 1 if direction == "hop_up" else int(new_state[m]) + 1)
            new_state[u] = str(int(new_state[u]) - 1 if direction == "hop_up" else int(new_state[u]) + 1)
            new_state[tm] = str(int(new_state[tm]) + 1 if direction == "hop_up" else int(new_state[tm]) - 1)
        elif direction in ("hop_right", "hop_left"):
            new_state[m] = str(int(new_state[m]) - 1 if direction == "hop_right" else int(new_state[m]) + 1)
            new_state[r] = str(int(new_state[r]) - 1 if direction == "hop_right" else int(new_state[r]) + 1)
            new_state[tl] = str(int(new_state[tl]) + 1 if direction == "hop_right" else int(new_state[tl]) - 1)
        else:
            raise ValueError(f"Unknown hopping direction: {direction}")
        return "".join(new_state)

    raise NotImplementedError("apply_hopping_spinS_fast currently supports S=1/2 and S=1.")


def generate_all_reachable_configs(
    initial_state: str,
    Lx: int,
    Ly: int,
    S: float,
    conserveenergy: bool,
    dynamics: str = "plaquette",
    m: float = 1.0,
    g: float = 1.0,
    ev: float = 1e-6,
    pbc: bool = False,
    progress: bool = False,
):
    """Generate all reachable configurations using breadth-first search."""
    clear_output = None
    if progress:
        try:
            from IPython.display import clear_output as ipython_clear_output

            clear_output = ipython_clear_output
        except ImportError:
            clear_output = None

    seen_states = {initial_state}
    all_states_with_cycle = {(initial_state, 0)}
    checked_states = set()
    queue = deque([(initial_state, 0)])

    step = 0
    max_queue = 1
    last_q = 0
    next_q_trigger = 1
    next_states_trigger = 2

    while queue:
        step += 1
        qlen = len(queue)
        nstates = len(seen_states)
        max_queue = max(max_queue, qlen)

        if progress:
            do_refresh = False
            if qlen >= next_q_trigger or qlen <= max(1, last_q // 2):
                do_refresh = True
                next_q_trigger = max(qlen + 1, int(qlen * 1.25) + 1)
            if nstates >= next_states_trigger:
                do_refresh = True
                while next_states_trigger <= nstates:
                    next_states_trigger *= 2
            if do_refresh:
                if clear_output is not None:
                    clear_output(wait=True)
                print(
                    f"[Step {step}] Queue={qlen} | AllStates={nstates} | "
                    f"Checked={len(checked_states)} | MaxQueue={max_queue}"
                )
                last_q = qlen

        current_state, current_cycle = queue.popleft()
        if current_state in checked_states:
            continue
        checked_states.add(current_state)

        if dynamics in ("plaquette", "both"):
            for _, indices, direction in plaquette_checker(current_state, Lx, Ly, S, conserveenergy, pbc):
                new_state = apply_plaquette_spinS_fast(current_state, indices, direction)
                if new_state not in seen_states:
                    seen_states.add(new_state)
                    all_states_with_cycle.add((new_state, current_cycle + 1))
                    queue.append((new_state, current_cycle + 1))

        if dynamics in ("hopping", "both"):
            for _, indices, direction in hopping_checker(current_state, initial_state, Lx, Ly, S, ev, m, g, pbc):
                new_state = apply_hopping_spinS_fast(current_state, indices, direction, S)
                if new_state not in seen_states:
                    seen_states.add(new_state)
                    all_states_with_cycle.add((new_state, current_cycle + 1))
                    queue.append((new_state, current_cycle + 1))

    if progress:
        if clear_output is not None:
            clear_output(wait=True)
        print("Finished BFS")
        print("Total reachable states:", len(seen_states))
        print("Total checked states:", len(checked_states))
        print("Maximum queue size:", max_queue)

    return all_states_with_cycle
