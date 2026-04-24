"""
E5: Extended barren plateau sweep including n=14,16.

This uses the same coefficient-space surrogate loss family as the
finishing scripts to produce an executable finite-size extension.

Writes:
  3_Results/barren_plateau_extended.json
"""

import json
import os
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "3_Results", "barren_plateau_extended.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def gradient_variance(n, j_round, samples=500, seed=42):
    rng = np.random.default_rng(seed + 100 * j_round + n)
    # coefficient-space surrogate "ansatz": normalized vector in R^n
    # random PSD matrix as stand-in laplacian
    a = rng.normal(size=(n, n))
    l = a.T @ a

    prev = []
    for _ in range(j_round):
        # deterministic pseudo-null vectors to mimic deflation penalties
        v = rng.normal(size=n)
        v /= np.linalg.norm(v) + 1e-12
        prev.append(v)

    mu = 5.0
    grads = []
    for _ in range(samples):
        x = rng.normal(size=n)
        x /= np.linalg.norm(x) + 1e-12
        eps = 1e-4
        e0 = np.zeros_like(x)
        e0[0] = 1.0

        def loss(z):
            z = z / (np.linalg.norm(z) + 1e-12)
            ray = float(z @ l @ z)
            pen = 0.0
            for pv in prev:
                pen += mu * float(np.dot(z, pv) ** 2)
            return ray + pen

        gp = loss(x + eps * e0)
        gm = loss(x - eps * e0)
        grads.append((gp - gm) / (2 * eps))
    return float(np.var(grads))


def fit_slope(ns, vals):
    ln = np.log(np.array(ns, dtype=float))
    lv = np.log(np.maximum(np.array(vals, dtype=float), 1e-30))
    a, b = np.polyfit(ln, lv, 1)
    # R^2
    pred = a * ln + b
    ss_res = float(np.sum((lv - pred) ** 2))
    ss_tot = float(np.sum((lv - np.mean(lv)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(r2)


def main():
    ns = [4, 6, 8, 10, 12, 14, 16]
    out = {"n_values": ns, "rounds": {}}
    for j in [0, 1, 2]:
        vals = [gradient_variance(n, j, samples=500, seed=42) for n in ns]
        slope_full, r2_full = fit_slope(ns, vals)
        slope_hi, r2_hi = fit_slope([12, 14, 16], vals[-3:])
        out["rounds"][str(j)] = {
            "variance_by_n": {str(n): float(v) for n, v in zip(ns, vals)},
            "slope_full_range": slope_full,
            "r2_full_range": r2_full,
            "slope_high_n_12_16": slope_hi,
            "r2_high_n_12_16": r2_hi,
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT}")
    for j in ["0", "1", "2"]:
        r = out["rounds"][j]
        print(
            f"j={j}: full slope={r['slope_full_range']:.3f}, "
            f"high-n slope={r['slope_high_n_12_16']:.3f}"
        )


if __name__ == "__main__":
    main()

