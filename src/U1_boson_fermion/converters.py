"""Converters between compact bitstrings and lattice array representations."""

from __future__ import annotations

import numpy as np


def converter_b2f(bi: str, lx: int, ly: int, s: float):
    """Convert a bitstring into matter, vertical-link, and horizontal-link arrays."""
    num = []
    links = []

    for i, char in enumerate(bi):
        if i % 3 == 0:
            num.append(float(char))
        else:
            links.append(char)

    hor = []
    ver = []
    for i in range(len(links) // 2):
        h = links[2 * i]
        v = links[2 * i + 1]
        if s == 1 / 2:
            hor.append(-0.5 if h == "0" else 0.5)
            ver.append(-0.5 if v == "0" else 0.5)
        else:
            hor.append(-1 if h == "0" else (1 if h == "2" else 0))
            ver.append(-1 if v == "0" else (1 if v == "2" else 0))

    hor = np.array(hor).reshape(lx, ly).T
    ver = np.array(ver).reshape(lx, ly).T
    num = np.array(num).reshape(lx, ly).T

    return num, ver, hor


def converter_f2s(Type: str, Input, s: float):
    """Convert binary strings to toolkit strings, or array triples to bitstrings."""
    if Type == "binary2string":
        output = []
        for i in range(len(Input) // 3):
            if s == 1 / 2:
                val1 = "0.5" if Input[3 * i + 1] == "1" else "-0.5"
                val2 = "0.5" if Input[3 * i + 2] == "1" else "-0.5"
            else:
                val1 = "-1" if Input[3 * i + 1] == "0" else ("1" if Input[3 * i + 1] == "2" else "0")
                val2 = "-1" if Input[3 * i + 2] == "0" else ("1" if Input[3 * i + 2] == "2" else "0")
            output.append(f"empty:{val1}:{val2}")
        return ":".join(output) + ":"

    if Type == "file2binary":
        two_flat = Input[1].T.flatten()
        three_flat = Input[2].T.flatten()
        one_flat = Input[0].T.flatten()
        binary = []
        for one, two, three in zip(one_flat, two_flat, three_flat):
            if s == 1 / 2:
                binary.append("0" if one == 0 else "1")
                binary.append("0" if two == -0.5 else "1")
                binary.append("0" if three == -0.5 else "1")
            else:
                binary.append("0" if one == 0 else "1")
                binary.append("0" if two == -1 else ("2" if two == 1 else "1"))
                binary.append("0" if three == -1 else ("2" if three == 1 else "1"))

        return "".join(binary)

    raise ValueError(f"Unknown Type: {Type}")


def converter_full(bi: str, ly: int, lx: int):
    """Spin-1/2 bitstring-to-array converter kept for notebook compatibility."""
    num = []
    links = []

    for i, char in enumerate(bi):
        if i % 3 == 0:
            num.append(float(char))
        else:
            links.append(char)

    hor = []
    ver = []
    for i in range(len(links) // 2):
        h = links[2 * i]
        v = links[2 * i + 1]
        hor.append(-0.5 if h == "0" else 0.5)
        ver.append(-0.5 if v == "0" else 0.5)

    hor = np.array(hor).reshape(lx, ly).T
    ver = np.array(ver).reshape(lx, ly).T
    num = np.array(num).reshape(lx, ly).T

    return num, ver, hor


def converter_half(Type: str, Input, x: int, y: int):
    """Half-filled lattice converter from the original notebook."""
    if Type == "binary2string":
        output = []
        for i in range(len(Input) // 3):
            if i // y % 2 == 0:
                val0 = "single" if int(Input[3 * i]) + ((-1) ** (i % 2 + 1) + 1) / 2 == 1 else "empty"
            else:
                val0 = "empty" if int(Input[3 * i]) + ((-1) ** (i % 2 + 1) + 1) / 2 == 1 else "single"
            val1 = "0.5" if Input[3 * i + 1] == "1" else "-0.5"
            val2 = "0.5" if Input[3 * i + 2] == "1" else "-0.5"
            output.append(f"{val0}:{val1}:{val2}")
        return ":".join(output) + ":"

    if Type == "file2binary":
        two_flat = Input[0].flatten()
        three_flat = Input[1].flatten()
        binary = []
        for two, three in zip(two_flat, three_flat):
            binary.append("0" if two == -0.5 else "1")
            binary.append("0" if three == -0.5 else "1")

        return "".join(binary)

    raise ValueError(f"Unknown Type: {Type}")


def inverse_converter_full(num, ver, hor) -> str:
    """Reconstruct a spin-1/2 bitstring from matter, vertical, and horizontal arrays."""
    ly, lx = num.shape
    chars = []

    for x in range(lx):
        for y in range(ly):
            matter_char = str(int(num[y, x]))
            hor_char = "0" if hor[y, x] == -0.5 else "1"
            ver_char = "0" if ver[y, x] == -0.5 else "1"
            chars.append(matter_char + hor_char + ver_char)

    return "".join(chars)
