import numpy as np

from spectral_lgt import ed, simulate_dynamics


def test_ed_can_return_values_only_or_vectors():
    H = np.diag([2.0, -1.0])

    vals = ed(H, return_eigenvectors=False)
    vals_with_vecs, vecs = ed(H)

    np.testing.assert_allclose(vals, [-1.0, 2.0])
    np.testing.assert_allclose(vals_with_vecs, [-1.0, 2.0])
    np.testing.assert_allclose(vecs.conj().T @ vecs, np.eye(2))


def test_spectrum_and_expm_dynamics_match_and_measure_operators():
    H = np.array([[0.0, 1.0], [1.0, 0.0]])
    psi0 = np.array([1.0, 0.0])
    times = np.array([0.0, 0.3, 0.7])
    operators = {"z": np.diag([1.0, -1.0])}

    spectrum = simulate_dynamics(
        H,
        psi0,
        times,
        operators=operators,
        method="spectrum",
        return_psi_t=True,
    )
    expm = simulate_dynamics(
        H,
        psi0,
        times,
        operators=operators,
        method="expm",
    )

    np.testing.assert_allclose(spectrum["psi_t"][-1], expm["psi_final"])
    np.testing.assert_allclose(spectrum["expectations"]["z"], np.cos(2 * times))
    assert "psi_t" not in expm
    assert "psi_final" in expm
