"""
verify_paper_claims.py
======================
Builds the FULL S&P 500 pipeline described in paper.tex from scratch,
using real market data, ripser, and the CE benchmark, then prints every
number the paper quotes so we can verify or correct them.

Issues to verify:
  Issue 5  — CE timing (39 ms vs 23 ms vs 16.4 s total)
  Issue 6  — β₁ peak at window 126, date 29 Feb 2008, β₁ = 17
  Issue 12 — β₀ "2×" ripser convention claim
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import time, json, os, sys, warnings
warnings.filterwarnings("ignore")

# ── 1. Download / load real S&P 500 data ──────────────────────────────────

def get_sp500_data(start="2003-01-02", end="2010-12-31"):
    """Download real S&P 500 daily closes from Yahoo Finance."""
    cache = os.path.join(os.path.dirname(__file__), "sp500_2003_2010.csv")
    if os.path.exists(cache):
        import pandas as pd
        df = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        print(f"Loaded cached data: {cache} ({len(df)} rows)")
        return df["Close"].values, df.index
    # Download
    import yfinance as yf
    import pandas as pd
    ticker = yf.Ticker("^GSPC")
    df = ticker.history(start=start, end=end, auto_adjust=True)
    df = df[["Close"]].dropna()
    df.to_csv(cache)
    print(f"Downloaded S&P 500 data -> {cache} ({len(df)} rows)")
    return df["Close"].values, df.index


# ── 2. Takens embedding ───────────────────────────────────────────────────

def takens_embed(series, tau, m):
    """Takens delay embedding.  Returns (n_pts, m) array."""
    n = len(series)
    n_pts = n - (m - 1) * tau
    return np.array([[series[t + k * tau] for k in range(m)]
                     for t in range(n_pts)])


# ── 3. VR complex + combinatorial Laplacian (dense) ──────────────────────

def vr_simplices(pts, eps, max_dim=2):
    """Return {k: list-of-tuples} for VR complex at scale eps."""
    D = squareform(pdist(pts))
    n = len(pts)
    simplices = {0: [(i,) for i in range(n)]}
    # edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                edges.append((i, j))
    simplices[1] = edges
    if max_dim >= 2:
        # triangles: clique enumeration on edge graph
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j); adj[j].add(i)
        tris = []
        for i, j in edges:
            common = adj[i] & adj[j]
            for k in common:
                tri = tuple(sorted([i, j, k]))
                tris.append(tri)
        simplices[2] = sorted(set(tris))
    return simplices


def boundary_matrix(simplices_k, simplices_km1):
    """Build boundary ∂_k as dense matrix."""
    nrow = len(simplices_km1)
    ncol = len(simplices_k)
    idx = {s: i for i, s in enumerate(simplices_km1)}
    B = np.zeros((nrow, ncol))
    for j, sigma in enumerate(simplices_k):
        for r in range(len(sigma)):
            face = sigma[:r] + sigma[r + 1:]
            sign = (-1) ** r
            if face in idx:
                B[idx[face], j] = sign
    return B


def combinatorial_laplacian(simplices, k):
    """Return Δ_k as dense array and its boundary matrices."""
    sk   = simplices.get(k, [])
    sk1  = simplices.get(k + 1, [])
    skm1 = simplices.get(k - 1, [])
    nk = len(sk)
    if nk == 0:
        return np.zeros((0, 0)), 0, 0

    # upper: ∂_{k+1} ∂_{k+1}^T
    if sk1:
        B_up = boundary_matrix(sk1, sk)
        L_up = B_up @ B_up.T
    else:
        L_up = np.zeros((nk, nk))

    # lower: ∂_k^T ∂_k
    if skm1:
        B_dn = boundary_matrix(sk, skm1)
        L_dn = B_dn.T @ B_dn
    else:
        L_dn = np.zeros((nk, nk))

    L = L_up + L_dn
    return L, len(sk1), len(skm1)


def betti_from_laplacian(L, c_delta=10.0):
    """Compute β = # eigenvalues below machine-eps threshold."""
    if L.shape[0] == 0:
        return 0, np.array([])
    evals = np.sort(np.real(la.eigvalsh(L)))
    eps_mach = np.finfo(float).eps
    nk = L.shape[0]
    spec_norm = np.max(np.abs(evals)) if len(evals) else 1.0
    delta = c_delta * eps_mach * nk * spec_norm
    delta = max(delta, 1e-8)  # floor
    beta = int(np.sum(evals < delta))
    return beta, evals


# ── 4. Ripser wrapper ────────────────────────────────────────────────────

def betti_via_ripser(pts, eps, maxdim=1):
    """Use ripser to compute Betti numbers at filtration value eps."""
    from ripser import ripser as rips
    # ripser computes full persistence; we read off Betti at the given eps
    result = rips(pts, maxdim=maxdim, thresh=eps * 1.01)  # slight pad
    bettis = {}
    for dim in range(maxdim + 1):
        dgm = result["dgms"][dim]
        # A feature is alive at eps if birth <= eps and (death > eps or death == inf)
        count = 0
        for birth, death in dgm:
            if birth <= eps and (death > eps or np.isinf(death)):
                count += 1
        bettis[dim] = count
    return bettis


# ── 5. Full pipeline ─────────────────────────────────────────────────────

def run_full_pipeline():
    print("=" * 70)
    print("VERIFICATION SCRIPT — checking paper.tex claims against real data")
    print("=" * 70)

    # ── Data ──
    prices, dates = get_sp500_data()
    log_returns = np.diff(np.log(prices))
    print(f"\nDataset: {len(prices)} trading days, {len(log_returns)} returns")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # ── Takens embedding ──
    tau, m = 13, 4
    embedded = takens_embed(log_returns, tau, m)
    print(f"\nTakens embedding: τ={tau}, m={m}")
    print(f"  Input length: {len(log_returns)}")
    print(f"  Embedded points: {len(embedded)} in R^{m}")

    # ── Sliding windows ──
    W = 500    # window width in trading days
    step = 8   # step size
    n_embed = len(embedded)
    # Each window has n_pts = W - (m-1)*tau embedded points
    n_pts_per_win = W - (m - 1) * tau
    print(f"\nSliding window: W={W}, step={step}")
    print(f"  Points per window: {n_pts_per_win}")

    # How many windows?
    starts = list(range(0, n_embed - n_pts_per_win + 1, step))
    n_windows = len(starts)
    print(f"  Number of windows: {n_windows}")

    # ── Filtration scale ──
    eps_star = 0.32
    print(f"  Filtration scale ε* = {eps_star}")

    # ── Process all windows ──
    beta0_ce   = np.zeros(n_windows, dtype=int)
    beta1_ce   = np.zeros(n_windows, dtype=int)
    beta0_rip  = np.zeros(n_windows, dtype=int)
    beta1_rip  = np.zeros(n_windows, dtype=int)
    n_edges    = np.zeros(n_windows, dtype=int)
    n_tris     = np.zeros(n_windows, dtype=int)
    sparsity   = np.zeros(n_windows)
    timing_eig = np.zeros(n_windows)
    timing_total = np.zeros(n_windows)
    win_dates_start = []
    win_dates_end   = []

    print(f"\nProcessing {n_windows} windows ...")
    t_pipeline_start = time.perf_counter()

    for w_idx, s in enumerate(starts):
        pts_raw = embedded[s : s + n_pts_per_win]
        # Standardise per dimension
        pts = (pts_raw - pts_raw.mean(axis=0)) / (pts_raw.std(axis=0) + 1e-12)

        # Dates for this window (approximate: use return-index offset)
        # The w-th embedded point corresponds to return index s + (m-1)*tau ... 
        # Actually the first return in the window starts at index s
        # The last return in the window is s + n_pts_per_win - 1 + (m-1)*tau
        first_return_idx = s
        last_return_idx  = s + n_pts_per_win - 1 + (m - 1) * tau
        # Return index i corresponds to date dates[i+1] (since return i = log(P_{i+1}/P_i))
        # But let's just use the price dates; return i is between date i and date i+1
        if last_return_idx + 1 < len(dates):
            win_dates_start.append(dates[first_return_idx])
            win_dates_end.append(dates[min(last_return_idx + 1, len(dates) - 1)])
        else:
            win_dates_start.append(dates[first_return_idx])
            win_dates_end.append(dates[-1])

        t0 = time.perf_counter()

        # Build VR complex
        smpx = vr_simplices(pts, eps_star, max_dim=2)
        ne = len(smpx.get(1, []))
        nt = len(smpx.get(2, []))
        n_edges[w_idx] = ne
        n_tris[w_idx]  = nt

        # CE: Laplacian eigendecomposition
        t_eig_start = time.perf_counter()
        if ne > 0:
            L1, _, _ = combinatorial_laplacian(smpx, k=1)
            b1, _ = betti_from_laplacian(L1)
            # Sparsity
            if L1.shape[0] > 0:
                nnz_per_row = np.count_nonzero(L1, axis=1).mean()
            else:
                nnz_per_row = 0
        else:
            b1 = 0
            nnz_per_row = 0

        # β₀ from 0-Laplacian
        L0, _, _ = combinatorial_laplacian(smpx, k=0)
        b0, _ = betti_from_laplacian(L0)
        timing_eig[w_idx] = time.perf_counter() - t_eig_start

        beta0_ce[w_idx] = b0
        beta1_ce[w_idx] = b1
        sparsity[w_idx] = nnz_per_row

        # Ripser
        rip = betti_via_ripser(pts, eps_star, maxdim=1)
        beta0_rip[w_idx] = rip[0]
        beta1_rip[w_idx] = rip[1]

        timing_total[w_idx] = time.perf_counter() - t0

        if (w_idx + 1) % 20 == 0 or w_idx == 0:
            print(f"  Window {w_idx+1:3d}/{n_windows}: "
                  f"edges={ne:4d} tris={nt:4d} "
                  f"β₀(CE)={b0} β₁(CE)={b1} "
                  f"β₀(rip)={rip[0]} β₁(rip)={rip[1]} "
                  f"t_eig={timing_eig[w_idx]*1e3:.1f}ms")

    t_pipeline_total = time.perf_counter() - t_pipeline_start

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # ── Basic stats ──
    print(f"\nTotal windows processed: {n_windows}")
    print(f"Points per window: {n_pts_per_win}")
    print(f"Edge counts n₁: min={n_edges.min()}, max={n_edges.max()}, "
          f"mean={n_edges.mean():.0f}, median={np.median(n_edges):.0f}")
    print(f"Triangle counts: min={n_tris.min()}, max={n_tris.max()}, "
          f"mean={n_tris.mean():.0f}")
    print(f"Laplacian sparsity (mean nnz/row): {sparsity[n_edges > 0].mean():.2f}, "
          f"max={sparsity.max():.1f}")

    # ── Issue 5: CE timing ──
    print(f"\n{'─'*50}")
    print("ISSUE 5 — CE timing")
    print(f"{'─'*50}")
    print(f"  Total pipeline time (all {n_windows} windows): {t_pipeline_total:.1f} s")
    print(f"  Mean eigendecomposition time/window: {timing_eig.mean()*1e3:.1f} ms")
    print(f"  Mean total time/window (VR+eig+ripser): {timing_total.mean()*1e3:.1f} ms")
    print(f"  Total eigendecomp time only: {timing_eig.sum():.1f} s")
    print(f"  Min eig time: {timing_eig.min()*1e3:.2f} ms, Max: {timing_eig.max()*1e3:.1f} ms")
    # What 16.4s / n_windows = implied per-window
    print(f"  If paper says 16.4s total → {16.4/n_windows*1e3:.1f} ms/window implied")

    # ── Issue 12: β₀ comparison CE vs ripser ──
    print(f"\n{'─'*50}")
    print("ISSUE 12 — β₀ '2×' ripser convention")
    print(f"{'─'*50}")
    b0_agree = np.sum(beta0_ce == beta0_rip)
    b0_ratio = beta0_rip.astype(float) / np.maximum(beta0_ce, 1)
    print(f"  β₀ exact agreement: {b0_agree}/{n_windows}")
    if b0_agree < n_windows:
        # Check if ripser consistently gives 2×
        diffs = np.where(beta0_ce != beta0_rip)[0]
        print(f"  Disagreements at windows: {diffs[:10]}...")
        for idx in diffs[:5]:
            print(f"    Window {idx}: CE={beta0_ce[idx]}, ripser={beta0_rip[idx]}, "
                  f"ratio={beta0_rip[idx]/max(beta0_ce[idx],1):.2f}")
        mean_ratio = b0_ratio[diffs].mean()
        print(f"  Mean ratio (ripser/CE) at disagreements: {mean_ratio:.2f}")
        is_2x = np.allclose(b0_ratio[diffs], 2.0, atol=0.1)
        print(f"  Is it consistently 2×?  {is_2x}")
    else:
        print(f"  β₀ agrees perfectly — the 2× claim is FALSE")
    print(f"  Mean β₀(CE)={beta0_ce.mean():.1f}, mean β₀(ripser)={beta0_rip.mean():.1f}")

    # ── β₁ comparison ──
    print(f"\n{'─'*50}")
    print("β₁ agreement CE vs ripser")
    print(f"{'─'*50}")
    b1_agree = np.sum(beta1_ce == beta1_rip)
    print(f"  β₁ exact agreement: {b1_agree}/{n_windows} "
          f"({'100%' if b1_agree == n_windows else f'{100*b1_agree/n_windows:.1f}%'})")
    if b1_agree < n_windows:
        diffs1 = np.where(beta1_ce != beta1_rip)[0]
        for idx in diffs1[:10]:
            print(f"    Window {idx}: CE={beta1_ce[idx]}, ripser={beta1_rip[idx]}")

    print(f"  β₁ range: [{beta1_ce.min()}, {beta1_ce.max()}]")
    print(f"  β₁ mean: {beta1_ce.mean():.2f}")

    # ── Issue 6: β₁ peak ──
    print(f"\n{'─'*50}")
    print("ISSUE 6 — β₁ peak window")
    print(f"{'─'*50}")
    peak_idx = np.argmax(beta1_ce)
    peak_b1 = beta1_ce[peak_idx]
    peak_start = win_dates_start[peak_idx]
    peak_end   = win_dates_end[peak_idx]
    print(f"  Peak β₁ = {peak_b1} at window index {peak_idx}")
    print(f"  Window start date: {peak_start.strftime('%Y-%m-%d')}")
    print(f"  Window end date:   {peak_end.strftime('%Y-%m-%d')}")
    print(f"  Paper claims: window 126, β₁ = 17, date '29 February 2008'")
    if peak_idx == 126:
        print(f"  ✓ Window index matches")
    else:
        print(f"  ✗ Window index MISMATCH (got {peak_idx}, paper says 126)")
    if peak_b1 == 17:
        print(f"  ✓ β₁ value matches")
    else:
        print(f"  ✗ β₁ MISMATCH (got {peak_b1}, paper says 17)")

    # Also check what window 126 actually is
    if 126 < n_windows:
        print(f"\n  Window 126 specifically:")
        print(f"    β₁(CE) = {beta1_ce[126]}")
        print(f"    Date start: {win_dates_start[126].strftime('%Y-%m-%d')}")
        print(f"    Date end:   {win_dates_end[126].strftime('%Y-%m-%d')}")

    # ── Table 1 representative windows ──
    print(f"\n{'─'*50}")
    print("Table 1 representative windows")
    print(f"{'─'*50}")
    # Find windows roughly matching the date ranges in the paper
    for label, target_start_yr, target_start_mo in [
        ("Jan 2007 – Dec 2008", 2007, 1),
        ("Mar 2007 – Feb 2009", 2007, 3),
        ("Jun 2007 – May 2009 (peak)", 2007, 6),
        ("Sep 2007 – Aug 2009", 2007, 9),
        ("Dec 2007 – Nov 2009", 2007, 12),
    ]:
        best_idx = None
        best_dist = 1e9
        for i, ds in enumerate(win_dates_start):
            dist = abs((ds.year - target_start_yr) * 12 + ds.month - target_start_mo)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            print(f"  {label}: win[{best_idx}] "
                  f"β₀(CE)={beta0_ce[best_idx]} β₁(CE)={beta1_ce[best_idx]} "
                  f"β₀(rip)={beta0_rip[best_idx]} β₁(rip)={beta1_rip[best_idx]} "
                  f"({win_dates_start[best_idx].strftime('%Y-%m-%d')} – "
                  f"{win_dates_end[best_idx].strftime('%Y-%m-%d')})")

    # ── Save verification results ──
    results = {
        "n_windows": int(n_windows),
        "n_pts_per_window": int(n_pts_per_win),
        "eps_star": eps_star,
        "n_edges_stats": {
            "min": int(n_edges.min()), "max": int(n_edges.max()),
            "mean": float(n_edges.mean()), "median": float(np.median(n_edges))
        },
        "n_tris_stats": {
            "min": int(n_tris.min()), "max": int(n_tris.max()),
            "mean": float(n_tris.mean())
        },
        "sparsity_mean_nnz_per_row": float(sparsity[n_edges > 0].mean()),
        "sparsity_max": float(sparsity.max()),
        "timing": {
            "total_pipeline_s": float(t_pipeline_total),
            "mean_eig_ms": float(timing_eig.mean() * 1e3),
            "total_eig_s": float(timing_eig.sum()),
            "mean_total_ms": float(timing_total.mean() * 1e3),
            "min_eig_ms": float(timing_eig.min() * 1e3),
            "max_eig_ms": float(timing_eig.max() * 1e3),
        },
        "beta0_ce_vs_ripser_agreement": int(b0_agree),
        "beta1_ce_vs_ripser_agreement": int(b1_agree),
        "beta1_range": [int(beta1_ce.min()), int(beta1_ce.max())],
        "beta1_mean": float(beta1_ce.mean()),
        "beta1_peak_window": int(peak_idx),
        "beta1_peak_value": int(peak_b1),
        "beta1_peak_date_start": peak_start.strftime('%Y-%m-%d'),
        "beta1_peak_date_end": peak_end.strftime('%Y-%m-%d'),
        "beta1_all_ce": beta1_ce.tolist(),
        "beta0_all_ce": beta0_ce.tolist(),
        "beta1_all_rip": beta1_rip.tolist(),
        "beta0_all_rip": beta0_rip.tolist(),
    }

    out_path = os.path.join(os.path.dirname(__file__), "3_Results", "verification_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_full_pipeline()
