"""Matplotlib plotting helpers for lattice states."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _default_site_values(Lx, Ly):
    site_val = np.zeros((Ly, Lx))
    for xx in range(Lx):
        for yy in range(Ly):
            site_val[yy][xx] = site_val[yy][xx] + (1 / 2) + (1 / 2) * (-1) ** (xx + yy + 1)
    return site_val


def plot_lattice_charge(ax, Lx, Ly, site_val, link_values_x, link_values_y, t, m, J, g, chi, pp, pe, s):
    """Plot charge representation on an existing Matplotlib axis."""
    x, y = np.meshgrid(np.arange(Lx), np.arange(Ly))
    if s == 1 / 2:
        link_values_x = np.abs(link_values_x - 1 / 2)
        link_values_y = np.abs(link_values_y - 1 / 2)
    if s == 1:
        link_values_x = link_values_x**2
        link_values_y = link_values_y**2

    site_norm = Normalize(vmin=-1, vmax=1)
    link_norm = Normalize(vmin=0, vmax=1)

    if len(site_val) == 0:
        site_val = _default_site_values(Lx, Ly)
    else:
        site_val = np.array(site_val, copy=True)

    segments_x = []
    colors_x = []
    for i in range(Ly):
        for j in range(Lx):
            segments_x.append([(j, i), (j + 1, i)])
            colors_x.append(link_values_x[i][j])
    lc_x = LineCollection(segments_x, cmap="Greys", array=np.array(colors_x), norm=link_norm, linewidths=2)
    ax.add_collection(lc_x)

    segments_y = []
    colors_y = []
    for i in range(Ly):
        for j in range(Lx):
            segments_y.append([(j, i), (j, i + 1)])
            colors_y.append(link_values_y[i][j])
    lc_y = LineCollection(segments_y, cmap="Greys", array=np.array(colors_y), norm=link_norm, linewidths=2)
    ax.add_collection(lc_y)

    for xx in range(Lx):
        for yy in range(Ly):
            site_val[yy][xx] = site_val[yy][xx] + (-1 / 2) + (1 / 2) * (-1) ** (xx + yy)

    site_scatter = ax.scatter(x, y, c=site_val.flatten(), cmap="coolwarm", norm=site_norm, s=100, zorder=2)

    ax.set_aspect("equal")
    ax.set_xticks(range(Lx))
    ax.set_yticks(range(Ly))
    ax.set_title(rf"t={t}, m = {m}, J = {J}, g = {g}, $\chi$ = {chi}")
    ax.scatter(pp[0], pp[1], facecolors="none", edgecolors="red", s=200, linewidths=2, zorder=3)
    ax.scatter(pe[0], pe[1], facecolors="none", edgecolors="blue", s=200, linewidths=2, zorder=3)

    return site_scatter, lc_x, lc_y


def plot_lattice_fermion(ax, Lx, Ly, site_val, link_values_x, link_values_y, t, m, J, g, chi, pp, pe, s):
    """Plot fermion representation on an existing Matplotlib axis."""
    x, y = np.meshgrid(np.arange(Lx), np.arange(Ly))
    site_norm = Normalize(vmin=0, vmax=1)
    if s == 1:
        link_norm = Normalize(vmin=-1, vmax=1)
    elif s == 1 / 2:
        link_norm = Normalize(vmin=-1 / 2, vmax=1 / 2)
    else:
        link_norm = Normalize(vmin=-s, vmax=s)

    if len(site_val) == 0:
        site_val = _default_site_values(Lx, Ly)

    segments_x = []
    colors_x = []
    for i in range(Ly):
        for j in range(Lx):
            segments_x.append([(j, i), (j + 1, i)])
            colors_x.append(link_values_x[i][j])
    lc_x = LineCollection(segments_x, cmap="coolwarm", array=np.array(colors_x), norm=link_norm, linewidths=2)
    ax.add_collection(lc_x)

    segments_y = []
    colors_y = []
    for i in range(Ly):
        for j in range(Lx):
            segments_y.append([(j, i), (j, i + 1)])
            colors_y.append(link_values_y[i][j])
    lc_y = LineCollection(segments_y, cmap="coolwarm", array=np.array(colors_y), norm=link_norm, linewidths=2)
    ax.add_collection(lc_y)

    site_scatter = ax.scatter(x, y, c=np.asarray(site_val).flatten(), cmap="Greys", norm=site_norm, s=100, zorder=2)

    ax.set_aspect("equal")
    ax.set_xticks(range(Lx))
    ax.set_yticks(range(Ly))
    if t != "/":
        ax.set_title(rf"t={t}, m = {m}, J = {J}, g = {g}, $\chi$ = {chi}")

    ax.scatter(pp[0], pp[1], facecolors="none", edgecolors="red", s=200, linewidths=2, zorder=3)
    ax.scatter(pe[0], pe[1], facecolors="none", edgecolors="blue", s=200, linewidths=2, zorder=3)

    return site_scatter, lc_x, lc_y
