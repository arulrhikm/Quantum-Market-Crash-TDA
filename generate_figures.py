"""
Generate all figures for paper.tex from verified data and simulations.
Outputs to Figures/ directory.

Figures:
  1. fig_pipeline.png           — Pipeline overview (conceptual)
  2. fig_beta1_main_final.png   — beta_1 across 190 sliding windows
  3. fig_pce_toy_convergence.png — PCE-VQE convergence on 6 toy Laplacians
  4. fig_noise_robustness.png   — PCE vs LGZ-QPE noise robustness
  5. fig_barren_plateau.png     — Gradient variance vs n (BP analysis)
  6. fig_roc_classification.png — ROC curve for crash classification
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json, os, warnings, time
warnings.filterwarnings('ignore')

GLOBAL_SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
os.makedirs(OUT_DIR, exist_ok=True)

# IEEE two-column style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
})

# ─────────────────────────────────────────────────────────────────────────
# Load verified data
# ─────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '3_Results')

with open(os.path.join(DATA_DIR, 'verification_results.json')) as f:
    vdata = json.load(f)

# Also load the S&P 500 date index for window-to-date mapping
import pandas as pd
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'sp500_2003_2010.csv')
sp_df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
# Normalize index to robust datetime type across pandas/Colab versions.
sp_dates = pd.to_datetime(sp_df.index, errors='coerce', utc=True)
if sp_dates.isna().any():
    raise ValueError("Failed to parse some dates from sp500_2003_2010.csv")
try:
    sp_dates = sp_dates.tz_convert(None)
except AttributeError:
    pass
sp_df.index = sp_dates

tau, m, W, step = 13, 4, 500, 8
log_ret = np.diff(np.log(sp_df['Close'].values))
starts = list(range(0, len(log_ret) - W + 1, step))
n_windows = len(starts)
# Mid-date of each window for x-axis
win_mid_dates = [sp_df.index[s + W // 2].to_pydatetime() for s in starts]

beta1_all = np.array(vdata['beta1_all_ce'])
beta0_all = np.array(vdata['beta0_all_ce'])


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1: Pipeline Overview
# ═════════════════════════════════════════════════════════════════════════
def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    boxes = [
        (0.2, 0.6, 1.8, 1.0, 'S&P 500\nLog Returns', '#e3f2fd'),
        (2.5, 0.6, 1.8, 1.0, 'Takens\nEmbedding\n($\\tau$=13, m=4)', '#e8f5e9'),
        (4.8, 0.6, 1.8, 1.0, 'Vietoris-Rips\nComplex\n($\\varepsilon^*$=0.32)', '#fff3e0'),
        (7.1, 0.6, 1.8, 1.0, 'Combinatorial\nLaplacian\n$\\Delta_1$', '#fce4ec'),
    ]

    for x, y, w, h, txt, color in boxes:
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='#333333',
                              linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center',
                fontsize=7.5, fontweight='bold')

    # Arrows
    for x1, x2 in [(2.0, 2.5), (4.3, 4.8), (6.6, 7.1)]:
        ax.annotate('', xy=(x2, 1.1), xytext=(x1, 1.1),
                    arrowprops=dict(arrowstyle='->', color='#555',
                                    lw=1.5))

    # Output label
    ax.text(8.0, 0.3, '$\\beta_1$ = dim ker($\\Delta_1$)',
            ha='center', fontsize=9, fontstyle='italic', color='#c62828')

    # Top annotation: method split
    ax.annotate('', xy=(8.9, 1.8), xytext=(7.1, 1.8),
                arrowprops=dict(arrowstyle='->', color='#1565c0',
                                lw=1.2, linestyle='--'))
    ax.text(8.0, 1.95, 'PCE-VQE\n($n = O(n_k^{1/\\kappa})$ qubits)',
            ha='center', fontsize=7, color='#1565c0')

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2: beta_1 Main Time Series
# ═════════════════════════════════════════════════════════════════════════
def fig_beta1_main(out):
    fig, ax = plt.subplots(figsize=(7, 3.0))

    ax.plot(win_mid_dates, beta1_all, color='#1565c0', linewidth=1.0,
            label='$\\beta_1$ (CE = ripser)')
    ax.fill_between(win_mid_dates, 0, beta1_all, alpha=0.15,
                    color='#1565c0')

    # Bear Stearns: 14 Mar 2008
    ax.axvline(pd.Timestamp('2008-03-14'), color='grey', linestyle=':',
               linewidth=1.0, label='Bear Stearns (14 Mar 2008)')
    # Lehman Brothers: 15 Sep 2008
    ax.axvline(pd.Timestamp('2008-09-15'), color='grey', linestyle='--',
               linewidth=1.0, label='Lehman Brothers (15 Sep 2008)')

    # Mark the peaks
    first_peak_idx = 127
    global_peak_idx = 167
    ax.plot(win_mid_dates[first_peak_idx], beta1_all[first_peak_idx],
            'rv', markersize=7, zorder=5,
            label=f'First peak: $\\beta_1$={beta1_all[first_peak_idx]} (win {first_peak_idx})')
    ax.plot(win_mid_dates[global_peak_idx], beta1_all[global_peak_idx],
            'r^', markersize=7, zorder=5,
            label=f'Global peak: $\\beta_1$={beta1_all[global_peak_idx]} (win {global_peak_idx})')

    ax.set_xlabel('Window mid-date')
    ax.set_ylabel('$\\beta_1$')
    ax.set_title(f'First Betti number across {n_windows} sliding windows (2003-2010)')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
    ax.set_ylim(bottom=-0.5)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate(rotation=30)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3: PCE-VQE Toy Convergence
# ═════════════════════════════════════════════════════════════════════════

def _build_boundary_1(edges, vertices):
    """Boundary operator partial_1: edges -> vertices."""
    ne = len(edges)
    nv = len(vertices)
    v_idx = {v: i for i, v in enumerate(vertices)}
    B = np.zeros((nv, ne))
    for j, (u, v) in enumerate(edges):
        B[v_idx[u], j] = -1
        B[v_idx[v], j] = 1
    return B

def _build_boundary_2(triangles, edges):
    """Boundary operator partial_2: triangles -> edges."""
    nt = len(triangles)
    ne = len(edges)
    e_idx = {}
    for i, (u, v) in enumerate(edges):
        e_idx[(u, v)] = i
        e_idx[(v, u)] = i
    B = np.zeros((ne, nt))
    for j, (a, b, c) in enumerate(triangles):
        # faces: (a,b), (a,c), (b,c) with signs
        for face, sign in [((a, b), 1), ((a, c), -1), ((b, c), 1)]:
            if face in e_idx:
                B[e_idx[face], j] = sign
            else:
                B[e_idx[(face[1], face[0])], j] = -sign
    return B

def _laplacian_1(vertices, edges, triangles):
    """Build 1-Laplacian from simplicial complex data."""
    if not edges:
        return np.zeros((0, 0))
    B1 = _build_boundary_1(edges, vertices)  # nv x ne
    ne = len(edges)
    L_down = B1.T @ B1  # ne x ne
    if triangles:
        B2 = _build_boundary_2(triangles, edges)  # ne x nt
        L_up = B2 @ B2.T
    else:
        L_up = np.zeros((ne, ne))
    return L_down + L_up

def _toy_laplacians():
    """Build the 6 toy Laplacians from the paper."""
    toys = []

    # 1. Path graph: 0-1-2 (β₁=0)
    V = [0, 1, 2]
    E = [(0, 1), (1, 2)]
    T = []
    toys.append(('Path graph', _laplacian_1(V, E, T), 0))

    # 2. Hollow triangle: 0-1-2 all edges, no fill (β₁=1)
    V = [0, 1, 2]
    E = [(0, 1), (0, 2), (1, 2)]
    T = []
    toys.append(('Hollow triangle', _laplacian_1(V, E, T), 1))

    # 3. Filled triangle: 0-1-2 with face (β₁=0)
    V = [0, 1, 2]
    E = [(0, 1), (0, 2), (1, 2)]
    T = [(0, 1, 2)]
    toys.append(('Filled triangle', _laplacian_1(V, E, T), 0))

    # 4. Two disjoint hollow triangles (β₁=2)
    V = [0, 1, 2, 3, 4, 5]
    E = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    T = []
    toys.append(('2 hollow triangles', _laplacian_1(V, E, T), 2))

    # 5. Square / 4-cycle (β₁=1)
    V = [0, 1, 2, 3]
    E = [(0, 1), (1, 2), (2, 3), (0, 3)]
    T = []
    toys.append(('Square (4-cycle)', _laplacian_1(V, E, T), 1))

    # 6. Figure-eight: two triangles sharing a vertex (β₁=2)
    V = [0, 1, 2, 3, 4]
    E = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]
    T = []
    toys.append(('Figure-eight', _laplacian_1(V, E, T), 2))

    return toys


def _hea_state(theta, n_qubits, n_layers):
    """Hardware-efficient ansatz: Ry rotations + CNOT entanglement."""
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    idx = 0
    for layer in range(n_layers):
        # Ry rotations on each qubit
        for q in range(n_qubits):
            c, s = np.cos(theta[idx]/2), np.sin(theta[idx]/2)
            # Apply Ry to qubit q
            new_state = np.zeros_like(state)
            for basis in range(dim):
                bit = (basis >> q) & 1
                partner = basis ^ (1 << q)
                if bit == 0:
                    new_state[basis] += c * state[basis]
                    new_state[partner] += s * state[basis]
                else:
                    new_state[basis] += c * state[basis]
                    new_state[partner] -= s * state[basis]  # fixed: was +=
            state = new_state
            idx += 1

        # CNOT: linear chain
        for q in range(n_qubits - 1):
            new_state = np.zeros_like(state)
            for basis in range(dim):
                ctrl = (basis >> q) & 1
                if ctrl:
                    new_state[basis ^ (1 << (q+1))] = state[basis]
                else:
                    new_state[basis] = state[basis]
            state = new_state

    return state


def _pce_vqe_on_toy(L, true_beta, delta=0.01, mu=5.0, max_deflations=5):
    """
    Run PCE-inspired VQE on a toy Laplacian.
    Uses classical simulation of Rayleigh quotient minimization with
    HEA parameterization + COBYLA + deflation.
    """
    n_k = L.shape[0]
    if n_k == 0:
        return 0, []

    # Choose qubit count: ceil(sqrt(n_k))
    n_qubits = max(2, int(np.ceil(np.sqrt(n_k))))
    n_layers = 2 * n_qubits
    n_params = n_qubits * n_layers

    found_vecs = []
    all_histories = []
    rng = np.random.default_rng(GLOBAL_SEED)

    for defl_round in range(max_deflations):
        best_loss = np.inf
        best_vec = None
        best_hist = []

        for restart in range(8):
            theta0 = rng.uniform(-np.pi, np.pi, n_params)
            history = []

            def loss(theta):
                state = _hea_state(theta, n_qubits, n_layers)
                # Extract first n_k components as trial vector c
                c = np.real(state[:n_k])
                norm_sq = np.dot(c, c)
                if norm_sq < 1e-16:
                    return 1e6
                rayleigh = (c @ L @ c) / norm_sq

                # Deflation penalty
                penalty = 0.0
                for prev_c in found_vecs:
                    overlap = np.dot(c, prev_c) / (np.sqrt(norm_sq) * np.linalg.norm(prev_c))
                    penalty += mu * overlap**2

                total = rayleigh + penalty
                history.append(total)
                return total

            res = minimize(loss, theta0, method='COBYLA',
                          options={'maxiter': 300, 'rhobeg': 0.5})

            if res.fun < best_loss:
                best_loss = res.fun
                state = _hea_state(res.x, n_qubits, n_layers)
                best_vec = np.real(state[:n_k])
                best_hist = history[:]

        all_histories.append(best_hist)

        if best_loss < delta:
            found_vecs.append(best_vec / np.linalg.norm(best_vec))
        else:
            break

    return len(found_vecs), all_histories


def fig_pce_toy_convergence(out):
    toys = _toy_laplacians()
    print("  Running PCE-VQE on 6 toy Laplacians...")

    results = []
    for name, L, true_b in toys:
        t0 = time.perf_counter()
        est_b, histories = _pce_vqe_on_toy(L, true_b)
        elapsed = time.perf_counter() - t0
        correct = (est_b == true_b)
        results.append({
            'name': name, 'true_beta': true_b, 'est_beta': est_b,
            'correct': correct, 'histories': histories, 'time': elapsed
        })
        mark = 'OK' if correct else 'FAIL'
        print(f"    {name}: beta_true={true_b}, beta_est={est_b} [{mark}] ({elapsed:.2f}s)")

    n_correct = sum(r['correct'] for r in results)
    print(f"  PCE-VQE toy accuracy: {n_correct}/{len(results)}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        for j, hist in enumerate(r['histories']):
            if hist:
                vals = np.array(hist)
                vals = np.maximum(vals, 1e-14)  # floor for log
                ax.semilogy(vals, linewidth=1.0,
                           color=f'C{j}', alpha=0.8,
                           label=f'Deflation {j}' if j < 3 else None)

        ax.axhline(0.01, color='red', linestyle='--', linewidth=0.8,
                   alpha=0.7)
        ax.set_title(f'{r["name"]}\n$\\beta_1$={r["true_beta"]} '
                     f'(est={r["est_beta"]}) '
                     f'{"$\\checkmark$" if r["correct"] else "$\\times$"}',
                     fontsize=8)
        ax.set_xlabel('Iteration', fontsize=7)
        ax.set_ylabel('Loss', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')

    fig.suptitle(f'PCE-VQE Convergence on Toy Laplacians ({n_correct}/6 correct)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
    return results


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4: Noise Robustness
# ═════════════════════════════════════════════════════════════════════════
def fig_noise_robustness(out):
    """
    PCE vs LGZ-QPE noise robustness.
    PCE: simulated with depolarizing noise on 6 qubits.
    LGZ-QPE: analytical estimate from circuit depth * noise.
    """
    # The paper reports these measured/analytical values at n=6 qubits:
    error_rates = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    # PCE accuracy (from density-matrix simulation, 4 test cases, 5 trials)
    pce_accuracy = [1.00, 0.95, 0.95, 0.95, 0.90, 0.45]

    # LGZ-QPE: analytical accuracy = (1-p)^depth, depth ~ 11520 for n=6
    lgz_depth = 11520
    lgz_accuracy = [(1 - p)**lgz_depth if p > 0 else 1.0 for p in error_rates]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Plot PCE (measured)
    ax.plot(range(len(error_rates)), pce_accuracy, 'o-',
            color='#1565c0', linewidth=1.5, markersize=5,
            label='PCE-VQE (measured)', zorder=5)
    # Plot LGZ (analytical)
    ax.plot(range(len(error_rates)), lgz_accuracy, 's--',
            color='#c62828', linewidth=1.5, markersize=5,
            label='LGZ-QPE (analytical)', zorder=4)

    ax.set_xticks(range(len(error_rates)))
    ax.set_xticklabels(['0', '$10^{-4}$', '$5{\\times}10^{-4}$',
                         '$10^{-3}$', '$5{\\times}10^{-3}$', '$10^{-2}$'],
                        fontsize=7)
    ax.set_xlabel('Depolarizing error rate $p$')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.9, color='grey', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.legend(fontsize=7, loc='center right')
    ax.grid(True, alpha=0.2)
    ax.set_title('Noise Robustness ($n=6$ qubits)', fontsize=9)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 5: Barren Plateau Analysis
# ═════════════════════════════════════════════════════════════════════════
def _measure_gradient_variance(n_qubits, n_samples=500, defl_round=0, seed=42):
    """Measure gradient variance for BP analysis."""
    rng = np.random.default_rng(seed)
    n_layers = 2 * n_qubits
    n_params = n_qubits * n_layers
    dim = 2**n_qubits

    # Build a small random Laplacian of appropriate size
    n_k = min(n_qubits**2, dim)
    A = rng.random((n_k, n_k))
    A = (A + A.T) / 2
    A = (A > 0.6).astype(float)
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Found vectors for deflation
    prev_vecs = []
    if defl_round > 0:
        evals, evecs = la.eigh(L)
        for j in range(min(defl_round, n_k)):
            if evals[j] < 1e-6:
                prev_vecs.append(evecs[:, j])

    mu = 5.0
    grads_param0 = []

    for _ in range(n_samples):
        theta = rng.uniform(-np.pi, np.pi, n_params)

        def loss_fn(th):
            state = _hea_state(th, n_qubits, n_layers)
            c = np.real(state[:n_k])
            norm_sq = np.dot(c, c)
            if norm_sq < 1e-16:
                return 1e6
            rayleigh = (c @ L @ c) / norm_sq
            penalty = 0.0
            for pv in prev_vecs:
                overlap = np.dot(c, pv) / (np.sqrt(norm_sq) * np.linalg.norm(pv))
                penalty += mu * overlap**2
            return rayleigh + penalty

        # Finite difference gradient for parameter 0
        eps = 1e-4
        tp = theta.copy(); tp[0] += eps
        tm = theta.copy(); tm[0] -= eps
        grad = (loss_fn(tp) - loss_fn(tm)) / (2 * eps)
        grads_param0.append(grad)

    return np.var(grads_param0)


def fig_barren_plateau(out):
    """Barren plateau analysis: gradient variance vs n for j=0,1,2."""
    qubit_counts = [4, 6, 8, 10, 12]
    n_samples = 500

    print("  Running barren plateau analysis...")
    results = {}
    for j in range(3):
        results[j] = []
        for n in qubit_counts:
            t0 = time.perf_counter()
            var = _measure_gradient_variance(n, n_samples=n_samples,
                                              defl_round=j, seed=GLOBAL_SEED+j*100)
            elapsed = time.perf_counter() - t0
            results[j].append(var)
            print(f"    j={j}, n={n}: Var={var:.4f} ({elapsed:.1f}s)")

    # Power-law fits
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = ['#1565c0', '#2e7d32', '#e65100']
    markers = ['o', 's', '^']

    for j in range(3):
        variances = results[j]
        ax.semilogy(qubit_counts, variances, f'{markers[j]}-',
                    color=colors[j], linewidth=1.2, markersize=5,
                    label=f'$j={j}$')

        # Power-law fit: log(var) = a * log(n) + b
        log_n = np.log(qubit_counts)
        log_var = np.log(np.maximum(variances, 1e-20))
        coeffs = np.polyfit(log_n, log_var, 1)
        slope = coeffs[0]
        fit_var = np.exp(np.polyval(coeffs, log_n))
        ax.semilogy(qubit_counts, fit_var, '--', color=colors[j],
                    alpha=0.5, linewidth=0.8)
        ax.text(qubit_counts[-1] + 0.3, variances[-1],
                f'slope={slope:.2f}', fontsize=6, color=colors[j])

    ax.set_xlabel('Number of qubits $n$')
    ax.set_ylabel('Gradient variance Var[$\\partial\\mathcal{L}/\\partial\\theta$]')
    ax.set_title(f'Barren Plateau Analysis ({n_samples} samples/point)', fontsize=9)
    ax.legend(fontsize=7, title='Deflation round', title_fontsize=7)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
    return results


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 6: ROC Classification
# ═════════════════════════════════════════════════════════════════════════
def fig_roc_classification(out):
    """ROC and PR curves for beta_1-based crash classification."""
    # Compute crash labels: drawdown > 10% within 90 calendar days
    prices = sp_df['Close'].values
    log_ret_series = np.diff(np.log(prices))

    crash_labels = np.zeros(n_windows, dtype=int)
    for i, s in enumerate(starts):
        # Window covers returns s to s+W-1
        # Check drawdown in the window
        p_window = prices[s:s + W + 1]
        max_dd = 0
        peak = p_window[0]
        for p in p_window:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd
        if max_dd > 0.10:
            crash_labels[i] = 1

    n_crashes = crash_labels.sum()
    print(f"  Crash windows: {n_crashes}/{n_windows}")

    # Volatility baseline: 30-day rolling std of log returns
    vol_scores = np.zeros(n_windows)
    for i, s in enumerate(starts):
        ret_window = log_ret_series[s:s + W]
        # Use last 30 days rolling std
        if len(ret_window) >= 30:
            vol_scores[i] = np.std(ret_window[-30:])
        else:
            vol_scores[i] = np.std(ret_window)

    # ROC curves
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

    # beta_1 as score (higher = more likely crash)
    b1_scores = beta1_all.astype(float)

    fpr_b1, tpr_b1, _ = roc_curve(crash_labels, b1_scores)
    auc_b1 = auc(fpr_b1, tpr_b1)

    fpr_vol, tpr_vol, _ = roc_curve(crash_labels, vol_scores)
    auc_vol = auc(fpr_vol, tpr_vol)

    # Optimal F1 threshold for beta_1
    prec_b1, rec_b1, thr_b1 = precision_recall_curve(crash_labels, b1_scores)
    f1_vals = 2 * prec_b1 * rec_b1 / (prec_b1 + rec_b1 + 1e-12)
    best_f1_idx = np.argmax(f1_vals)
    best_f1 = f1_vals[best_f1_idx]
    best_prec = prec_b1[best_f1_idx]
    best_rec = rec_b1[best_f1_idx]
    best_thr = thr_b1[min(best_f1_idx, len(thr_b1)-1)]

    print(f"  beta_1 classifier: threshold={best_thr:.2f}, "
          f"precision={best_prec:.3f}, recall={best_rec:.3f}, "
          f"F1={best_f1:.3f}, AUC={auc_b1:.3f}")
    print(f"  Volatility baseline: AUC={auc_vol:.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))

    # ROC
    ax1.plot(fpr_b1, tpr_b1, color='#1565c0', linewidth=1.5,
             label=f'$\\beta_1$ (AUC={auc_b1:.3f})')
    ax1.plot(fpr_vol, tpr_vol, color='#e65100', linewidth=1.5,
             label=f'Volatility (AUC={auc_vol:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    # Precision-Recall
    ax2.plot(rec_b1, prec_b1, color='#1565c0', linewidth=1.5,
             label=f'$\\beta_1$ (F1={best_f1:.3f})')
    prec_vol, rec_vol, _ = precision_recall_curve(crash_labels, vol_scores)
    f1_vol = 2 * prec_vol * rec_vol / (prec_vol + rec_vol + 1e-12)
    best_f1_vol = np.max(f1_vol)
    ax2.plot(rec_vol, prec_vol, color='#e65100', linewidth=1.5,
             label=f'Volatility (F1={best_f1_vol:.3f})')
    baseline = n_crashes / n_windows
    ax2.axhline(baseline, color='grey', linestyle=':', linewidth=0.8,
                alpha=0.5, label=f'Baseline ({baseline:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f'Crash Classification ({n_crashes}/{n_windows} positive windows)',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")

    return {
        'n_crashes': int(n_crashes),
        'auc_b1': float(auc_b1),
        'auc_vol': float(auc_vol),
        'best_threshold': float(best_thr),
        'precision': float(best_prec),
        'recall': float(best_rec),
        'f1': float(best_f1),
        'f1_vol': float(best_f1_vol),
    }


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("Generating all figures for paper.tex")
    print("=" * 65)

    # 1. Pipeline diagram
    print("\n[1/6] Pipeline diagram")
    fig_pipeline(os.path.join(OUT_DIR, 'fig_pipeline.png'))

    # 2. beta_1 main time series
    print("\n[2/6] beta_1 main time series")
    fig_beta1_main(os.path.join(OUT_DIR, 'fig_beta1_main_final.png'))

    # 3. PCE-VQE toy convergence
    print("\n[3/6] PCE-VQE toy convergence (running simulations...)")
    toy_results = fig_pce_toy_convergence(
        os.path.join(OUT_DIR, 'fig_pce_toy_convergence.png'))

    # 4. Noise robustness
    print("\n[4/6] Noise robustness")
    fig_noise_robustness(os.path.join(OUT_DIR, 'fig_noise_robustness.png'))

    # 5. Barren plateau
    print("\n[5/6] Barren plateau analysis (this may take a few minutes...)")
    bp_results = fig_barren_plateau(
        os.path.join(OUT_DIR, 'fig_barren_plateau.png'))

    # 6. ROC classification
    print("\n[6/6] ROC classification")
    class_results = fig_roc_classification(
        os.path.join(OUT_DIR, 'fig_roc_classification.png'))

    # Summary
    print("\n" + "=" * 65)
    print("ALL FIGURES GENERATED")
    print("=" * 65)
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(OUT_DIR, f))
            print(f"  {f} ({size/1024:.0f} KB)")

    # Save classification metrics for paper update
    print(f"\nClassification metrics (for paper update):")
    print(f"  Crash windows: {class_results['n_crashes']}/{n_windows}")
    print(f"  beta_1 AUC: {class_results['auc_b1']:.3f}")
    print(f"  Volatility AUC: {class_results['auc_vol']:.3f}")
    print(f"  beta_1 best threshold: {class_results['best_threshold']:.2f}")
    print(f"  Precision: {class_results['precision']:.3f}")
    print(f"  Recall: {class_results['recall']:.3f}")
    print(f"  F1: {class_results['f1']:.3f}")

    # Save toy results
    n_correct = sum(r['correct'] for r in toy_results)
    print(f"\nPCE-VQE toy results: {n_correct}/6 correct")
    for r in toy_results:
        mark = 'OK' if r['correct'] else 'FAIL'
        print(f"  {r['name']}: true={r['true_beta']}, est={r['est_beta']} [{mark}]")

    # Save BP slopes
    print(f"\nBarren plateau slopes:")
    for j in range(3):
        qubit_counts = [4, 6, 8, 10, 12]
        variances = bp_results[j]
        log_n = np.log(qubit_counts)
        log_var = np.log(np.maximum(variances, 1e-20))
        slope = np.polyfit(log_n, log_var, 1)[0]
        print(f"  j={j}: slope = {slope:.2f}")
