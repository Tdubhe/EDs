import numpy as np

from spectral_lgt import converter_b2f, converter_f2s, inverse_converter_full


def test_spin_half_round_trip():
    num = np.array([[0, 1], [1, 0]])
    hor = np.array([[-0.5, 0.5], [0.5, -0.5]])
    ver = np.array([[0.5, -0.5], [-0.5, 0.5]])

    bitstring = converter_f2s("file2binary", [num, hor, ver], s=1 / 2)
    out_num, out_ver, out_hor = converter_b2f(bitstring, lx=2, ly=2, s=1 / 2)

    np.testing.assert_array_equal(out_num, num)
    np.testing.assert_array_equal(out_ver, ver)
    np.testing.assert_array_equal(out_hor, hor)
    assert inverse_converter_full(out_num, out_ver, out_hor) == bitstring


def test_spin_one_file_to_binary():
    num = np.array([[0, 1], [1, 0]])
    hor = np.array([[-1, 0], [1, -1]])
    ver = np.array([[1, -1], [0, 1]])

    bitstring = converter_f2s("file2binary", [num, hor, ver], s=1)
    out_num, out_ver, out_hor = converter_b2f(bitstring, lx=2, ly=2, s=1)

    np.testing.assert_array_equal(out_num, num)
    np.testing.assert_array_equal(out_ver, ver)
    np.testing.assert_array_equal(out_hor, hor)
