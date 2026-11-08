"""Synthetic fields for the N2 null model and for the known-answer tests."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def gaussian_field(x, y, bbox, xi, rng, res=None):
    """Zero-mean unit-variance Gaussian random field at (x, y) with correlation exp(-r^2 / xi^2).

    xi <= 0 gives independent N(0, 1) values. The correlation equals 1/e at r = xi, the same definition
    as the estimated correlation length (probes.corr_length).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if xi is None or xi <= 0:
        return rng.standard_normal(x.size)
    sig = xi / 2.0
    res = res or max(min(sig / 3.0, 100.0), 10.0)
    pad = 3 * xi
    x0, y0, x1, y1 = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad
    nxr = int(np.ceil((x1 - x0) / res)) + 1
    nyr = int(np.ceil((y1 - y0) / res)) + 1
    g = ndimage.gaussian_filter(rng.standard_normal((nyr, nxr)), sigma=sig / res, mode="wrap")
    g = (g - g.mean()) / g.std()
    return ndimage.map_coordinates(g, [(y - y0) / res, (x - x0) / res], order=1, mode="nearest")


def stationary_flows(x, y, bbox, xi, sigma, mu, n, rng):
    """N2: n log-normal realisations exp(mu + sigma Z) on the given layout (columns)."""
    return np.column_stack([np.exp(mu + sigma * gaussian_field(x, y, bbox, xi, rng)) for _ in range(n)])


def monocentric_flows(x, y, bbox, xi, sigma, mu, weight, n, rng, centre=None, r0=None):
    """Stationary field mixed with a radial gradient around one centre (known-answer tests, X9)."""
    cx, cy = centre if centre is not None else ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    r0 = r0 or 0.25 * min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    g = -np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) / r0
    g = (g - g.mean()) / g.std()
    cols = []
    for _ in range(n):
        z = np.sqrt(1 - weight) * gaussian_field(x, y, bbox, xi, rng) + np.sqrt(weight) * g
        z = (z - z.mean()) / z.std()
        cols.append(np.exp(mu + sigma * z))
    return np.column_stack(cols)
