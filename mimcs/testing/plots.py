"""Graphical inspection of MCMC output.

These plots are not part of automated pass/fail --- they are saved to disk for a human
to inspect, which for MCMC often reveals problems (poor mixing, missed modes, wrong
geometry) that summary statistics miss. The two most useful are:

* :func:`trace_plot` --- each coordinate against iteration, to eyeball mixing.
* :func:`pair_plot` --- a corner-style grid of marginal histograms (diagonal) and
  pairwise scatter (lower triangle), with the exact reference overlaid when available
  so the sampler's cloud can be compared to ground truth directly.
* :func:`step_size_plot` --- the step-size trajectory over warmup. Step size co-adapts
  with almost every other quantity, so its convergence is a useful proxy for the
  convergence of adaptation as a whole.

Uses a non-interactive backend so it works headless.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _thin(x: np.ndarray, max_points: int) -> np.ndarray:
    if len(x) <= max_points:
        return x
    step = int(np.ceil(len(x) / max_points))
    return x[::step]


def trace_plot(samples: np.ndarray, labels, path: str, *, title: str | None = None,
               max_points: int = 2000):
    samples = np.atleast_2d(np.asarray(samples, float))
    if samples.ndim == 1:
        samples = samples[:, None]
    d = samples.shape[1]
    thinned = _thin(samples, max_points)
    idx = np.linspace(0, len(samples) - 1, len(thinned)).astype(int)

    fig, axes = plt.subplots(d, 1, figsize=(9, 1.8 * d + 0.6), sharex=True, squeeze=False)
    for j in range(d):
        ax = axes[j, 0]
        ax.plot(idx, thinned[:, j], lw=0.5, color="C0")
        ax.set_ylabel(labels[j])
    axes[-1, 0].set_xlabel("iteration")
    fig.suptitle(title or "trace")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def step_size_plot(step_sizes, path: str, *, title: str | None = None):
    """Plot the warmup step-size trajectory (log scale, since the update is on log-step).

    Returns ``None`` without writing if there is no step-size history (e.g. a sampler
    that does not carry a ``step_size``)."""
    s = np.asarray(step_sizes, float)
    if s.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(1, len(s) + 1), s, lw=0.8, color="C1")
    ax.set_yscale("log")
    ax.set_xlabel("warmup iteration")
    ax.set_ylabel("step size")
    ax.set_title(title or "warmup step size")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def clip_threshold_plot(log_clip, path: str, *, title: str | None = None):
    """Plot the score-mass gradient-clip threshold (``_sm_log_clip``) over warmup.

    The threshold is an online log-scale quantile of the gradient norm that calibrates the
    clipping; tracking it shows whether the clipping settles. Returns ``None`` without
    writing if there is no history."""
    s = np.asarray(log_clip, float)
    if s.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(1, len(s) + 1), s, lw=0.8, color="C2")
    ax.set_xlabel("warmup iteration")
    ax.set_ylabel("log clip threshold")
    ax.set_title(title or "score-mass clip threshold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def pair_plot(samples: np.ndarray, labels, path: str, *, reference: np.ndarray | None = None,
              title: str | None = None, max_points: int = 3000):
    """Corner-style grid: marginal histograms on the diagonal, pairwise scatter below.

    If ``reference`` is given (e.g. exact draws), it is drawn underneath in grey so the
    sampler's cloud (colour) can be compared against ground truth.
    """
    samples = np.atleast_2d(np.asarray(samples, float))
    if samples.ndim == 1:
        samples = samples[:, None]
    d = samples.shape[1]
    S = _thin(samples, max_points)
    R = _thin(np.asarray(reference, float), max_points) if reference is not None else None

    fig, axes = plt.subplots(d, d, figsize=(2.4 * d + 1, 2.4 * d + 1), squeeze=False)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                if R is not None:
                    ax.hist(R[:, i], bins=40, density=True, color="0.7",
                            alpha=0.8, label="reference")
                ax.hist(S[:, i], bins=40, density=True, histtype="step",
                        color="C0", lw=1.5, label="sampler")
                if i == 0 and R is not None:
                    ax.legend(fontsize=7, loc="upper right")
            elif i > j:
                if R is not None:
                    ax.scatter(R[:, j], R[:, i], s=4, color="0.7", alpha=0.4)
                ax.scatter(S[:, j], S[:, i], s=4, color="C0", alpha=0.4)
            else:
                ax.axis("off")
            if i == d - 1:
                ax.set_xlabel(labels[j])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
    fig.suptitle(title or "pairwise samples")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
