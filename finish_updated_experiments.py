"""
Run remaining paper-finishing experiments (E1, E1b, E6, E7).

Outputs:
  - 3_Results/classification_chronological.json
  - 3_Results/classification_ood.json
  - 3_Results/hardware_estimate.json
  - 3_Results/hyperparameters.json
  - Figures/fig_roc_classification_split.png
  - REPRODUCE.md
"""

import json
import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.circuit.library import TwoLocal
from qiskit.transpiler import CouplingMap
from qiskit import transpile


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "3_Results")
FIG_DIR = os.path.join(ROOT, "Figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


TAU = 13
M = 4
W = 500
STEP = 8
EPS_STAR = 0.32


def takens_embed(series: np.ndarray, tau: int = TAU, m: int = M) -> np.ndarray:
    n_pts = len(series) - (m - 1) * tau
    return np.array([[series[t + k * tau] for k in range(m)] for t in range(n_pts)])


def compute_window_starts(n_returns: int, w: int = W, step: int = STEP):
    return list(range(0, n_returns - w + 1, step))


def forward_drawdown_labels(prices: np.ndarray, dates: pd.Series, starts, w: int = W):
    """
    Forward-looking label used by the experiment plan:
    1 if price drops >10% from the window endpoint within next 90 calendar days.
    """
    labels = np.zeros(len(starts), dtype=int)
    for i, s in enumerate(starts):
        end_idx = s + w
        if end_idx >= len(prices):
            continue
        end_price = prices[end_idx]
        end_date = dates.iloc[end_idx]
        mask = (dates > end_date) & (dates <= end_date + pd.Timedelta(days=90))
        future = prices[mask.to_numpy()]
        if future.size == 0:
            continue
        labels[i] = int(np.min(future) <= 0.90 * end_price)
    return labels


def vol_scores(log_returns: np.ndarray, starts, w: int = W):
    scores = np.zeros(len(starts), dtype=float)
    for i, s in enumerate(starts):
        rw = log_returns[s : s + w]
        scores[i] = float(np.std(rw[-30:])) if len(rw) >= 30 else float(np.std(rw))
    return scores


def best_threshold_from_train(y, scores):
    uniq = np.unique(scores)
    best_t, best_f1 = uniq[0], -1.0
    for t in uniq:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def metrics(y_true, scores, threshold):
    if len(y_true) == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "auc": float("nan")}
    y_pred = (scores >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, scores))
    else:
        out["auc"] = float("nan")
    return out


def run_e1():
    vpath = os.path.join(RESULTS_DIR, "verification_results.json")
    with open(vpath, "r", encoding="utf-8") as f:
        v = json.load(f)
    beta1 = np.array(v["beta1_all_ce"], dtype=float)

    df = pd.read_csv(os.path.join(ROOT, "sp500_2003_2010.csv"), parse_dates=["Date"])
    prices = df["Close"].to_numpy()
    dates = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    log_ret = np.diff(np.log(prices))
    starts = compute_window_starts(len(log_ret))
    start_dates = dates.iloc[starts].reset_index(drop=True)

    y = forward_drawdown_labels(prices, dates, starts)
    vol = vol_scores(log_ret, starts)

    train_mask = start_dates.dt.year <= 2006
    test_mask = start_dates.dt.year >= 2007

    th_b1 = best_threshold_from_train(y[train_mask], beta1[train_mask])
    th_vol = best_threshold_from_train(y[train_mask], vol[train_mask])

    out = {
        "threshold_beta_1": th_b1,
        "threshold_vol": th_vol,
        "counts": {
            "total_windows": int(len(y)),
            "total_positive": int(np.sum(y)),
            "train_windows": int(np.sum(train_mask)),
            "test_windows": int(np.sum(test_mask)),
            "train_positive": int(np.sum(y[train_mask])),
            "test_positive": int(np.sum(y[test_mask])),
        },
        "beta1_train": metrics(y[train_mask], beta1[train_mask], th_b1),
        "beta1_test": metrics(y[test_mask], beta1[test_mask], th_b1),
        "vol_train": metrics(y[train_mask], vol[train_mask], th_vol),
        "vol_test": metrics(y[test_mask], vol[test_mask], th_vol),
    }

    # ROC split figure
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, mask, scores, color, ls in [
        ("beta1 train", train_mask, beta1, "#1565c0", "--"),
        ("beta1 test", test_mask, beta1, "#1565c0", "-"),
        ("vol train", train_mask, vol, "#e65100", "--"),
        ("vol test", test_mask, vol, "#e65100", "-"),
    ]:
        ym = y[mask]
        sm = scores[mask]
        if len(np.unique(ym)) > 1:
            fpr, tpr, _ = roc_curve(ym, sm)
            auc = roc_auc_score(ym, sm)
            ax.plot(fpr, tpr, linestyle=ls, color=color, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Chronological Split ROC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_roc_classification_split.png"), dpi=300)
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "classification_chronological.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def build_window_beta1(prices: np.ndarray, tau=TAU, m=M, w=W, step=STEP, eps=EPS_STAR):
    log_ret = np.diff(np.log(prices))
    emb = takens_embed(log_ret, tau=tau, m=m)
    n_pts = w - (m - 1) * tau
    starts = list(range(0, len(emb) - n_pts + 1, step))
    b1 = []
    for s in starts:
        cloud = emb[s : s + n_pts]
        cloud = (cloud - cloud.mean(axis=0)) / (cloud.std(axis=0) + 1e-12)
        r = ripser(cloud, maxdim=1, thresh=eps * 1.01)
        dgm = r["dgms"][1]
        c = 0
        for birth, death in dgm:
            if birth <= eps and (death > eps or np.isinf(death)):
                c += 1
        b1.append(c)
    return np.array(b1, dtype=float), starts


def run_e1b(th_beta, th_vol):
    episodes = {
        # download_start, eval_start, eval_end
        "covid_2020": ("2018-01-01", "2019-06-01", "2020-12-31"),
        "rate_cycle_2022": ("2020-01-01", "2022-01-01", "2023-06-30"),
    }
    out = {"thresholds_fixed_from_e1": {"beta_1": th_beta, "vol": th_vol}, "episodes": {}}

    for name, (download_start, eval_start, eval_end) in episodes.items():
        df = yf.download("^GSPC", start=download_start, end=eval_end, auto_adjust=True, progress=False)
        if df.empty or "Close" not in df:
            out["episodes"][name] = {"error": "download_failed"}
            continue
        prices = np.asarray(df["Close"]).astype(float).reshape(-1)
        dates = pd.to_datetime(df.index, utc=True).tz_convert(None)
        if len(prices) < (W + 120):
            out["episodes"][name] = {"error": "insufficient_length", "n_prices": int(len(prices))}
            continue
        b1, starts = build_window_beta1(prices)
        log_ret = np.diff(np.log(prices))
        y_all = forward_drawdown_labels(prices, pd.Series(dates), starts, w=W)
        vol = vol_scores(log_ret, starts, w=W)
        # With W=500, window starts are much earlier than the crisis period.
        # Evaluate OOD by window end date lying in the target episode.
        end_dates = pd.Series(dates).iloc[[s + W for s in starts]].reset_index(drop=True)
        mask = (end_dates >= pd.Timestamp(eval_start)) & (end_dates <= pd.Timestamp(eval_end))
        y = y_all[mask.to_numpy()]
        b1m = b1[mask.to_numpy()]
        volm = vol[mask.to_numpy()]
        if len(y) == 0:
            out["episodes"][name] = {"error": "no_windows_after_filter", "n_windows": 0}
            continue
        out["episodes"][name] = {
            "n_windows": int(len(y)),
            "n_positive": int(np.sum(y)),
            "beta1": metrics(y, b1m, th_beta),
            "vol": metrics(y, volm, th_vol),
        }

    with open(os.path.join(RESULTS_DIR, "classification_ood.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_e6():
    n = 16
    reps = 32
    ansatz = TwoLocal(n, rotation_blocks=["ry", "rz"], entanglement_blocks="cx", entanglement="linear", reps=reps)
    coupling = CouplingMap.from_line(n)
    t = transpile(
        ansatz,
        basis_gates=["cx", "sx", "rz"],
        coupling_map=coupling,
        optimization_level=3,
    )
    counts = t.count_ops()
    n_cx = int(counts.get("cx", 0))
    n_1q = int(sum(v for k, v in counts.items() if k != "cx"))
    depth = int(t.depth())

    e2 = 3e-3
    e1 = 2e-4
    f = (1 - e2) ** n_cx * (1 - e1) ** n_1q

    n_shots_basis = 10000
    G = 3
    n_iter = 200
    n_rounds = 2
    total_shots = n_shots_basis * G * n_iter * n_rounds

    out = {
        "n_qubits": n,
        "reps": reps,
        "transpiled_depth": depth,
        "n_cx": n_cx,
        "n_1q": n_1q,
        "eps_2q": e2,
        "eps_1q": e1,
        "per_shot_fidelity_estimate": f,
        "shots_per_basis": n_shots_basis,
        "commuting_bases_G": G,
        "iterations_per_round": n_iter,
        "rounds": n_rounds,
        "total_shot_budget": total_shots,
    }
    with open(os.path.join(RESULTS_DIR, "hardware_estimate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_e7(ch, ood, hw):
    hyper = {
        "takens": {"tau": TAU, "m": M},
        "windowing": {"W": W, "step": STEP},
        "vr": {"epsilon_star": EPS_STAR},
        "classification": {
            "label_definition": "forward-looking: drop >10% from window endpoint within next 90 calendar days",
            "threshold_beta_1": ch["threshold_beta_1"],
            "threshold_vol": ch["threshold_vol"],
        },
        "optimizer": {"name": "COBYLA", "mu": 5.0, "delta": 0.01},
    }
    with open(os.path.join(RESULTS_DIR, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(hyper, f, indent=2)

    reproduce = """# Reproduce Key Paper Outputs

## Core verification
- `python verify_paper_claims.py`
- Output: `3_Results/verification_results.json`

## Figure generation
- `python generate_figures.py`
- Outputs: `Figures/fig_*.png`

## Finishing experiments (E1, E1b, E6, E7)
- `python finish_updated_experiments.py`
- Outputs:
  - `3_Results/classification_chronological.json`
  - `3_Results/classification_ood.json`
  - `3_Results/hardware_estimate.json`
  - `3_Results/hyperparameters.json`
  - `Figures/fig_roc_classification_split.png`
"""
    with open(os.path.join(ROOT, "REPRODUCE.md"), "w", encoding="utf-8") as f:
        f.write(reproduce)


def main():
    ch = run_e1()
    ood = run_e1b(ch["threshold_beta_1"], ch["threshold_vol"])
    hw = run_e6()
    run_e7(ch, ood, hw)
    print("Finished E1, E1b, E6, E7.")
    print("E1 total positives:", ch["counts"]["total_positive"])
    print("E1 beta1 test:", ch["beta1_test"])
    print("E6:", hw)


if __name__ == "__main__":
    main()

