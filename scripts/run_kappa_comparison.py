"""
E4: kappa=2 vs kappa=3 comparison on n_k=64, beta1=2 benchmark.

Writes:
  results/kappa_comparison.json
"""

import json
import os
import numpy as np
from scipy.optimize import minimize


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "kappa_comparison.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def build_two_disjoint_32_cycles_laplacian():
    """
    Build 1-Laplacian for two disjoint 32-edge cycles:
      n_k = 64 edges, beta1 = 2.
    """
    n_cycles = 2
    n_per = 32
    edges = []
    v_offset = 0
    for _ in range(n_cycles):
        verts = list(range(v_offset, v_offset + n_per))
        for i in range(n_per):
            edges.append((verts[i], verts[(i + 1) % n_per]))
        v_offset += n_per

    n_vertices = n_cycles * n_per
    nv = n_vertices
    ne = len(edges)
    vidx = {v: i for i, v in enumerate(range(n_vertices))}

    b1 = np.zeros((nv, ne))
    for j, (u, v) in enumerate(edges):
        b1[vidx[u], j] = -1.0
        b1[vidx[v], j] = 1.0

    # no 2-simplices in pure cycle graph
    l1 = b1.T @ b1
    return l1


def hea_state(theta, n_qubits, n_layers):
    dim = 2 ** n_qubits
    s = np.zeros(dim, dtype=float)
    s[0] = 1.0
    p = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            a = theta[p]
            p += 1
            c, sn = np.cos(a / 2), np.sin(a / 2)
            ns = np.zeros_like(s)
            for b in range(dim):
                bit = (b >> q) & 1
                partner = b ^ (1 << q)
                if bit == 0:
                    ns[b] += c * s[b]
                    ns[partner] += sn * s[b]
                else:
                    ns[b] += c * s[b]
                    ns[partner] -= sn * s[b]
            s = ns
    nrm = np.linalg.norm(s)
    return s / nrm if nrm > 1e-12 else s


def run_variant(lap, n_qubits, n_layers, seeds=1, delta=0.05, mu=5.0, rounds=2, maxiter=120):
    n_k = lap.shape[0]
    dim = 2 ** n_qubits
    # Surrogate PCE decoding map from compressed quantum state to n_k coefficients.
    # This allows n_k > 2^n while preserving deterministic reproducibility.
    dec_rng = np.random.default_rng(1234 + n_qubits * 17 + n_layers)
    decode = dec_rng.normal(0.0, 1.0, size=(n_k, dim))
    decode /= (np.linalg.norm(decode, axis=1, keepdims=True) + 1e-12)
    n_params = n_qubits * n_layers
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(42 + seed)
        found = []
        iters = []
        succ = 0
        for _r in range(rounds):
            best_fun = np.inf
            best_vec = None
            best_iter = maxiter
            for _restart in range(1):
                t0 = rng.uniform(-np.pi, np.pi, n_params)

                def loss(t):
                    st = hea_state(t, n_qubits, n_layers)
                    c = decode @ st
                    nrm = float(np.dot(c, c))
                    if nrm < 1e-16:
                        return 1e6
                    ray = float(c @ lap @ c) / nrm
                    pen = 0.0
                    for prev in found:
                        ov = float(np.dot(c, prev)) / (np.sqrt(nrm) * np.linalg.norm(prev))
                        pen += mu * ov * ov
                    return ray + pen

                res = minimize(loss, t0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
                if res.fun < best_fun:
                    best_fun = float(res.fun)
                    st = hea_state(res.x, n_qubits, n_layers)
                    best_vec = st[:n_k].copy()
                    best_iter = int(res.nfev)
            iters.append(best_iter)
            if best_fun < delta:
                succ += 1
                found.append(best_vec / np.linalg.norm(best_vec))
        out.append({"seed": seed, "success_rounds": succ, "iters": iters})
        print(f"n={n_qubits} seed={seed} success={succ}/2 mean_iter={np.mean(iters):.1f}")

    mean_iter = float(np.mean([np.mean(x["iters"]) for x in out]))
    mean_succ = float(np.mean([x["success_rounds"] for x in out]))
    return {
        "n_qubits": n_qubits,
        "layers": n_layers,
        "depth_proxy_2n2": int(2 * n_qubits * n_qubits),
        "param_proxy_2nL": int(2 * n_qubits * n_layers),
        "mean_iterations_to_converge": mean_iter,
        "mean_deflation_success_rounds": mean_succ,
        "trials": out,
    }


def main():
    lap = build_two_disjoint_32_cycles_laplacian()
    data = {
        "benchmark": {"n_k": int(lap.shape[0]), "beta1_true": 2},
        "kappa2": run_variant(lap, n_qubits=8, n_layers=16),
        "kappa3": run_variant(lap, n_qubits=5, n_layers=10),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {OUT}")
    print("kappa2:", data["kappa2"]["mean_iterations_to_converge"], data["kappa2"]["mean_deflation_success_rounds"])
    print("kappa3:", data["kappa3"]["mean_iterations_to_converge"], data["kappa3"]["mean_deflation_success_rounds"])


if __name__ == "__main__":
    main()

