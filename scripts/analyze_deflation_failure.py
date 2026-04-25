"""
Analyze deflation failure at beta1=4 (toy case).

Writes:
  results/deflation_failure_analysis.json
"""

import json
import os
import numpy as np
from scipy.optimize import minimize


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "deflation_failure_analysis.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def build_four_disjoint_cycles_laplacian(n_per=8):
    edges = []
    v = 0
    for _ in range(4):
        vs = list(range(v, v + n_per))
        for i in range(n_per):
            edges.append((vs[i], vs[(i + 1) % n_per]))
        v += n_per
    nv, ne = 4 * n_per, len(edges)
    b1 = np.zeros((nv, ne))
    for j, (u, w) in enumerate(edges):
        b1[u, j] = -1
        b1[w, j] = 1
    l1 = b1.T @ b1
    return l1


def state(theta, n, layers):
    dim = 2 ** n
    s = np.zeros(dim)
    s[0] = 1.0
    p = 0
    for _ in range(layers):
        for q in range(n):
            a = theta[p]
            p += 1
            c, sn = np.cos(a / 2), np.sin(a / 2)
            ns = np.zeros_like(s)
            for b in range(dim):
                bit = (b >> q) & 1
                pr = b ^ (1 << q)
                if bit == 0:
                    ns[b] += c * s[b]
                    ns[pr] += sn * s[b]
                else:
                    ns[b] += c * s[b]
                    ns[pr] -= sn * s[b]
            s = ns
    nrm = np.linalg.norm(s)
    return s / nrm if nrm > 1e-12 else s


def run_case(lap, n=6, layers=12, mu=5.0, rounds=5, seeds=8, maxiter=350, delta=0.01):
    n_k = lap.shape[0]
    n_params = n * layers
    results = []
    for seed in range(seeds):
        rng = np.random.default_rng(100 + seed)
        found = []
        succ = 0
        losses = []
        for _ in range(rounds):
            best = 1e9
            best_vec = None
            for _r in range(4):
                t0 = rng.uniform(-np.pi, np.pi, n_params)

                def loss(t):
                    st = state(t, n, layers)
                    c = st[:n_k]
                    nrm = np.dot(c, c)
                    if nrm < 1e-16:
                        return 1e6
                    ray = float(c @ lap @ c) / float(nrm)
                    pen = 0.0
                    for prev in found:
                        ov = float(np.dot(c, prev)) / (np.sqrt(nrm) * np.linalg.norm(prev))
                        pen += mu * ov * ov
                    return ray + pen

                r = minimize(loss, t0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
                if r.fun < best:
                    best = float(r.fun)
                    st = state(r.x, n, layers)
                    best_vec = st[:n_k].copy()
            losses.append(best)
            if best < delta:
                succ += 1
                found.append(best_vec / np.linalg.norm(best_vec))
        results.append({"seed": seed, "success_rounds": succ, "best_losses": losses})
        print(f"mu={mu} layers={layers} seed={seed} success={succ}/5")
    succ_mean = float(np.mean([r["success_rounds"] for r in results]))
    fail_rate = float(np.mean([r["success_rounds"] < 4 for r in results]))
    return {"n": n, "layers": layers, "mu": mu, "succ_mean": succ_mean, "fail_rate_lt4": fail_rate, "runs": results}


def main():
    lap = build_four_disjoint_cycles_laplacian()
    out = {"beta1_true": 4, "sweeps": []}
    for mu in [1.0, 3.0, 5.0, 8.0]:
        out["sweeps"].append(run_case(lap, n=6, layers=12, mu=mu))
    for layers in [8, 12, 16]:
        out["sweeps"].append(run_case(lap, n=6, layers=layers, mu=5.0))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT}")
    for s in out["sweeps"]:
        print("mu", s["mu"], "layers", s["layers"], "succ_mean", s["succ_mean"], "fail<4", s["fail_rate_lt4"])


if __name__ == "__main__":
    main()

