import numpy as np

from spectral_lgt import (
    apply_plaquette_spinS_fast,
    basis_generator,
    converter_f2s,
    ham,
    mag,
    mag_y,
    trotter_ham,
)


def test_spin_one_hamiltonian_respects_explicit_zero_J_when_g_nonzero():
    state = list("010000001000")
    indices = [2, 4, 1, 8]
    connected_state = apply_plaquette_spinS_fast(state, indices, "forward")
    state = "".join(state)

    basis_names = [state, connected_state]
    basis = {name: [idx, idx] for idx, name in enumerate(basis_names)}

    explicit_zero = ham(
        basis,
        basis_names,
        Lx=2,
        Ly=2,
        m=0,
        J=0,
        t=0,
        g=1,
        ev=np.inf,
        fermion=False,
        S=1,
    )
    auto_j = ham(
        basis,
        basis_names,
        Lx=2,
        Ly=2,
        m=0,
        J=0,
        t=0,
        g=1,
        ev=np.inf,
        fermion=False,
        S=1,
        auto_J_from_g=True,
    )

    assert explicit_zero[0, 1] == 0
    assert auto_j[0, 1] == -1


def test_trotter_pieces_sum_to_hamiltonian():
    Lx = Ly = 2
    S = 1
    m = 0.3
    J = 0.2
    t = 1.1
    g = 0.7

    num = np.array([[0, 1], [1, 0]], dtype=int)
    hor = np.ones((Ly, Lx), dtype=int)
    ver = np.ones((Ly, Lx), dtype=int)
    initial = converter_f2s("file2binary", [num, hor, ver], s=S)

    basis, basis_names = basis_generator(
        initial,
        Lx=Lx,
        Ly=Ly,
        m=m,
        g=g,
        ev=np.inf,
        dynamics="both",
        S=S,
    )

    full_ham = ham(
        basis,
        basis_names,
        Lx=Lx,
        Ly=Ly,
        m=m,
        J=J,
        t=t,
        g=g,
        ev=np.inf,
        fermion=True,
        S=S,
    )
    trotter_sum = sum(
        trotter_ham(
            basis,
            basis_names,
            Lx=Lx,
            Ly=Ly,
            m=m,
            J=J,
            t=t,
            g=g,
            ev=np.inf,
            fermion=True,
            S=S,
        )
    )

    assert (trotter_sum - full_ham).nnz == 0


def test_magnetization_operators_use_explicit_J_and_are_hermitian():
    state = list("010000001000")
    indices = [2, 4, 1, 8]
    connected_state = apply_plaquette_spinS_fast(state, indices, "forward")
    state = "".join(state)

    basis_names = [state, connected_state]
    basis = {name: [idx, idx] for idx, name in enumerate(basis_names)}

    mx = mag(
        basis,
        basis_names,
        Lx=2,
        Ly=2,
        J=0.2,
        S=1,
    )
    my = mag_y(
        basis,
        basis_names,
        Lx=2,
        Ly=2,
        J=0.2,
        S=1,
    )

    assert mx[0, 1] == 0.2
    assert mx[1, 0] == 0.2
    assert my[0, 1] == 0.2j
    assert my[1, 0] == -0.2j
    assert (mx - mx.getH()).nnz == 0
    assert (my - my.getH()).nnz == 0
