"""
=============================================================================
Quantum Computation of Topological Invariants for Financial Time-Series Analysis
=============================================================================
MSEF 2026 — Complete Implementation with Rigorous Benchmarking

AUDIT FINDINGS (Phase 1):
--------------------------
Original project deficiencies:
1. QPE used uniform superposition then only counted p(|000...>), which conflates
   the all-zero eigenvalue probability with random coincidence. This is WRONG:
   kernel dimension estimation requires threshold-based eigenvalue counting.
2. No classical baseline was rigorously defined or compared against.
3. No statistical benchmarking across multiple Laplacian sizes.
4. Complexity analysis section was absent.
5. Limitations were not formally stated.
6. Betti number "estimate" β = p(0)*2^q is only valid if eigenstates are
   uniformly sampled — in practice they are not.

This file corrects ALL deficiencies.

Author: MSEF 2026 Submission
Date: February 2026
=============================================================================
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.special import comb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import warnings
import time
import json
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# RANDOM SEED CONTROL — reproducibility requirement
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
rng = np.random.default_rng(GLOBAL_SEED)


# =============================================================================
# SECTION 1: CLASSICAL TDA INFRASTRUCTURE
# =============================================================================

class VietorisRipsComplex:
    """
    Build a Vietoris-Rips simplicial complex from a point cloud.

    Given a finite metric space (X, d) and parameter epsilon > 0, the
    Vietoris-Rips complex VR(X, epsilon) has:
      - k-simplices = subsets {x_0,...,x_k} with d(x_i, x_j) <= epsilon forall i,j.

    This is a standard construction in persistent homology (Edelsbrunner & Harer 2010).
    """

    def __init__(self, points: np.ndarray, epsilon: float, max_dim: int = 2):
        self.points = np.array(points)
        self.n = len(points)
        self.epsilon = epsilon
        self.max_dim = max_dim
        self.simplices = {k: [] for k in range(max_dim + 1)}
        self._build()

    def _pairwise_distances(self) -> np.ndarray:
        """Compute all pairwise Euclidean distances O(n^2 * d)."""
        diff = self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def _build(self):
        D = self._pairwise_distances()
        # 0-simplices: vertices
        self.simplices[0] = [(i,) for i in range(self.n)]
        # k-simplices for k >= 1
        for k in range(1, self.max_dim + 1):
            for combo in combinations(range(self.n), k + 1):
                # All pairwise distances in this set must be <= epsilon
                valid = all(D[i, j] <= self.epsilon
                            for i, j in combinations(combo, 2))
                if valid:
                    self.simplices[k].append(combo)

    def num_simplices(self, k: int) -> int:
        return len(self.simplices.get(k, []))

    def chain_group_dim(self, k: int) -> int:
        return self.num_simplices(k)


class CombinorialLaplacian:
    """
    Construct and analyze the combinatorial (Hodge) Laplacian.

    Definition: For a simplicial complex K,
      Δ_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k  ∈ R^{C_k × C_k}

    where ∂_k : C_k → C_{k-1} is the boundary operator.

    The k-th Betti number is:
      β_k = dim(ker Δ_k) = (number of zero eigenvalues of Δ_k)

    This follows from the Hodge decomposition theorem:
      H_k(K) ≅ ker(Δ_k)

    Reference: Eckmann 1944, Horak & Jost 2013.
    """

    def __init__(self, vr_complex: VietorisRipsComplex, k: int):
        self.vrc = vr_complex
        self.k = k
        self.simplices_k = vr_complex.simplices.get(k, [])
        self.simplices_k1 = vr_complex.simplices.get(k + 1, [])
        self.n_k = len(self.simplices_k)
        self.n_k1 = len(self.simplices_k1)
        self.L = self._compute_laplacian()
        self._eigenvalues = None

    def _boundary_operator(self) -> np.ndarray:
        """
        Build the boundary matrix ∂_{k+1}: C_{k+1} → C_k.
        Column j = boundary of (k+1)-simplex j, expressed in the basis of k-simplices.
        """
        if self.n_k == 0 or self.n_k1 == 0:
            return np.zeros((self.n_k, self.n_k1))

        # Index map: simplex tuple → row index
        simplex_index = {s: i for i, s in enumerate(self.simplices_k)}
        B = np.zeros((self.n_k, self.n_k1))

        for j, sigma in enumerate(self.simplices_k1):
            # boundary(σ) = Σ_i (-1)^i * face_i(σ)
            for i in range(len(sigma)):
                face = sigma[:i] + sigma[i + 1:]  # remove i-th vertex
                if face in simplex_index:
                    B[simplex_index[face], j] = (-1) ** i
        return B

    def _compute_laplacian(self) -> np.ndarray:
        """Construct the combinatorial Laplacian Δ_k = ∂_{k+1}∂_{k+1}^T + ∂_k^T ∂_k."""
        if self.n_k == 0:
            return np.zeros((0, 0))

        # Upper part: ∂_{k+1} ∂_{k+1}^T  (dim: n_k × n_k)
        B_up = self._boundary_operator()
        L_up = B_up @ B_up.T

        # Lower part: ∂_k^T ∂_k
        # We need boundary ∂_k: C_k → C_{k-1}
        lower_complex = CombinorialLaplacian.__new__(CombinorialLaplacian)
        lower_complex.vrc = self.vrc
        lower_complex.k = self.k - 1
        lower_complex.simplices_k = self.vrc.simplices.get(self.k - 1, [])
        lower_complex.simplices_k1 = self.simplices_k
        lower_complex.n_k = len(lower_complex.simplices_k)
        lower_complex.n_k1 = self.n_k
        B_down = lower_complex._boundary_operator()  # shape (n_{k-1}, n_k)
        L_down = B_down.T @ B_down  # shape (n_k, n_k)

        return L_up + L_down

    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:
            if self.L.size == 0:
                self._eigenvalues = np.array([])
            else:
                self._eigenvalues = np.real(la.eigvalsh(self.L))
        return self._eigenvalues

    def betti_number_classical(self, tol: float = 1e-8) -> int:
        """
        Compute β_k = dim(ker Δ_k) by counting eigenvalues <= tol.
        This is the EXACT classical method. Complexity: O(n_k^3).
        """
        if len(self.eigenvalues) == 0:
            return 0
        return int(np.sum(self.eigenvalues <= tol))

    def matrix_size(self) -> int:
        return self.n_k


def compute_betti_classical(points: np.ndarray, epsilon: float,
                             max_dim: int = 2, tol: float = 1e-8) -> dict:
    """
    Full classical pipeline: point cloud → Rips complex → Laplacian → Betti numbers.

    Returns dict with keys 'beta_k', 'laplacian_k', 'eigenvalues_k', 'timing'.
    """
    t0 = time.perf_counter()
    vrc = VietorisRipsComplex(points, epsilon, max_dim)
    results = {}
    for k in range(max_dim + 1):
        lap = CombinorialLaplacian(vrc, k)
        results[f'beta_{k}'] = lap.betti_number_classical(tol)
        results[f'laplacian_{k}'] = lap.L
        results[f'eigenvalues_{k}'] = lap.eigenvalues
        results[f'matrix_size_{k}'] = lap.matrix_size()
    results['timing_classical'] = time.perf_counter() - t0
    return results


# =============================================================================
# SECTION 2: QUANTUM PHASE ESTIMATION (CORRECTED)
# =============================================================================

class QuantumPhaseEstimation:
    """
    Simulated Quantum Phase Estimation for estimating β_k = dim(ker Δ_k).

    CORRECTED ALGORITHM (fixing the original project's flaw):
    ----------------------------------------------------------
    The original code estimated β_k ≈ p(|0...0>) * 2^q, which is incorrect.
    The probability of measuring |0...0> in QPE is related to the weight of
    zero-eigenvalue eigenstates in the initial state, NOT directly to β_k.

    CORRECT method:
    1. For each basis vector |j> in C_k (j = 0,...,n_k-1):
       a. Run QPE with initial eigenstate register = |j>
       b. QPE measures phase φ satisfying U|ψ> = e^{2πiφ}|ψ>
       c. Convert measured phase → eigenvalue estimate λ_est = φ * λ_max / δ
       d. If |λ_est| <= τ (threshold), this basis vector has zero-eigenvalue component
    2. Repeat for random sample of basis vectors
    3. Estimate β_k = n_k * (fraction with near-zero eigenvalue estimate)

    Mathematical justification:
    - If |j> = Σ_m c_{jm} |ψ_m> (eigendecomposition),
      then QPE measures phase corresponding to eigenvalue λ_m with prob |c_{jm}|^2
    - If Δ_k |ψ_m> = 0 (harmonic form), then λ_m = 0 → phase φ = 0 → measured |0...0>
    - A basis vector |j> lies in ker(Δ_k) iff all eigenvalues with nonzero overlap are 0
    - We estimate β_k by sampling basis vectors and computing the fraction in ker(Δ_k)

    Precision analysis:
    - q precision qubits → phase resolution δφ = 1/2^q → eigenvalue resolution δλ = λ_max/2^q
    - Threshold τ should satisfy: 0 < τ << min nonzero eigenvalue

    Reference: Harrow, Hassidim, Lloyd (2009); Lloyd, Garnerone, Zanardi (2016).
    """

    def __init__(self, laplacian: np.ndarray, n_precision_qubits: int = 5,
                 delta: float = 0.5, shots: int = 1024,
                 seed: int = GLOBAL_SEED):
        self.L = laplacian
        self.n = laplacian.shape[0]
        self.q_prec = n_precision_qubits
        self.delta = delta  # scaling: U = exp(i * delta * L / lambda_max)
        self.shots = shots
        self.rng = np.random.default_rng(seed)

        if self.n == 0:
            self.U = np.zeros((0, 0), dtype=complex)
            self.lambda_max = 0.0
            return

        # Construct unitary U = exp(i * delta * L / lambda_max)
        # This maps L's zero eigenvalues to U's +1 eigenvalues (phase = 0)
        # Non-zero eigenvalues map to distinct phases
        evals = np.real(la.eigvalsh(self.L))
        self.lambda_max = max(np.max(np.abs(evals)), 1e-10)
        H_normalized = (delta / self.lambda_max) * self.L
        # Matrix exponential of skew-Hermitian = unitary
        self.U = la.expm(1j * H_normalized)
        # Verify unitarity
        assert np.allclose(self.U @ self.U.conj().T,
                           np.eye(self.n), atol=1e-8), "U must be unitary"

    def _qft_matrix(self, m: int) -> np.ndarray:
        """Discrete Quantum Fourier Transform matrix of size 2^m × 2^m."""
        N = 2 ** m
        omega = np.exp(2j * np.pi / N)
        indices = np.arange(N)
        return (1 / np.sqrt(N)) * omega ** np.outer(indices, indices)

    def _simulate_qpe_single(self, psi_init: np.ndarray) -> float:
        """
        Simulate QPE for a single initial state psi_init.

        Returns: estimated phase φ in [0, 1).

        The QPE circuit:
        1. |0>^q |ψ> → (QFT|0>)^q |ψ>  via Hadamards
        2. Apply controlled-U^{2^j} for j = 0,...,q-1
        3. Apply IQFT to precision register
        4. Measure precision register → binary fraction φ

        In simulation, we decompose |ψ> = Σ c_m |ψ_m> and compute the
        measurement probabilities analytically.
        """
        if self.n == 0:
            return 0.0

        M = 2 ** self.q_prec  # precision register size
        psi = psi_init / np.linalg.norm(psi_init)

        # Eigendecompose U: U = V diag(e^{2πiφ_m}) V†
        evals_U, evecs_U = la.eig(self.U)
        phases_true = np.angle(evals_U) / (2 * np.pi)  # φ_m ∈ (-0.5, 0.5]
        phases_true = phases_true % 1.0  # map to [0, 1)

        # Coefficients: c_m = <ψ_m|ψ>
        coeffs = evecs_U.conj().T @ psi  # shape (n,)

        # QPE measurement probabilities:
        # P(k) = |Σ_m c_m * (1/√M) * Σ_{j=0}^{M-1} e^{2πi j (φ_m - k/M)}|^2
        # For large M this is approximately a delta function at k/M = φ_m
        # For finite M it's a Dirichlet kernel
        probs = np.zeros(M)
        for k in range(M):
            amplitude = 0.0 + 0j
            for m in range(self.n):
                phi_m = phases_true[m]
                # Inner sum: Σ_{j=0}^{M-1} e^{2πij(φ_m - k/M)}
                alpha = phi_m - k / M
                if abs(alpha % 1.0) < 1e-12 or abs(alpha % 1.0 - 1.0) < 1e-12:
                    inner_sum = M
                else:
                    inner_sum = (1 - np.exp(2j * np.pi * M * alpha)) / (
                                 1 - np.exp(2j * np.pi * alpha))
                amplitude += coeffs[m] * inner_sum / np.sqrt(M)
            probs[k] = abs(amplitude) ** 2

        # Normalize (numerical errors)
        total = probs.sum()
        if total > 1e-12:
            probs /= total

        # Sample a measurement outcome
        k_meas = self.rng.choice(M, p=probs)
        phi_meas = k_meas / M
        return phi_meas

    def estimate_betti_threshold(self, n_samples: int = 30,
                                  threshold_tau: float = None) -> dict:
        """
        Estimate β_k using threshold-based eigenvalue detection.

        Algorithm (corrected):
        1. For each of n_samples random basis states |j>:
           a. Run QPE → measured phase φ
           b. Convert to eigenvalue: λ_est = φ * λ_max / δ (for φ < 0.5)
              or λ_est = (1-φ) * λ_max / δ (for φ >= 0.5, negative eigenvalue branch)
           c. If λ_est <= τ, mark as zero
        2. β_k_est = n_k * (count of zeros / n_samples)

        Note: The Laplacian is PSD, so all true eigenvalues >= 0.
        Phase φ ≈ 0 corresponds to λ ≈ 0 (in kernel).
        Phase φ close to 1 also corresponds to near-zero eigenvalue (phase wraps).

        Parameters:
        -----------
        n_samples: int — number of random basis states to sample
        threshold_tau: float — eigenvalue threshold; if None, use λ_max / 2^(q_prec-1)

        Returns: dict with estimate, std error, all measurements, raw phases
        """
        if self.n == 0:
            return {'betti_estimate': 0, 'betti_raw': 0.0,
                    'zero_count': 0, 'n_samples': 0,
                    'phases': [], 'lambda_estimates': [],
                    'threshold_used': 0.0}

        if threshold_tau is None:
            # Threshold should be set between 0 and the smallest nonzero eigenvalue.
            # QPE phase resolution: δφ = 1/2^q_prec
            # Corresponding eigenvalue resolution: δλ = δφ * 2π * λ_max / δ
            # Set threshold at 0.4 * phase_resolution_as_eigenvalue
            phase_resolution = (1.0 / (2 ** self.q_prec)) * 2 * np.pi * self.lambda_max / self.delta
            threshold_tau = 0.4 * phase_resolution

        zero_count = 0
        phases = []
        lambda_estimates = []

        for _ in range(n_samples):
            # Random basis vector
            j = self.rng.integers(0, self.n)
            psi_j = np.zeros(self.n, dtype=complex)
            psi_j[j] = 1.0

            phi = self._simulate_qpe_single(psi_j)
            phases.append(phi)

            # Convert phase to eigenvalue estimate
            # U = exp(i * δ * L / λ_max)
            # eigenvalue of U: e^{i * δ * λ / λ_max} where λ is eigenvalue of L
            # φ = δ * λ / (2π * λ_max)  → λ = φ * 2π * λ_max / δ
            # But we wrap phases: use min(φ, 1-φ) for near-zero detection
            phi_wrapped = min(phi, 1.0 - phi)
            lambda_est = phi_wrapped * 2 * np.pi * self.lambda_max / self.delta
            lambda_estimates.append(lambda_est)

            if lambda_est <= threshold_tau:
                zero_count += 1

        betti_raw = self.n * zero_count / n_samples
        betti_estimate = int(round(betti_raw))

        return {
            'betti_estimate': betti_estimate,
            'betti_raw': betti_raw,
            'zero_count': zero_count,
            'n_samples': n_samples,
            'phases': phases,
            'lambda_estimates': lambda_estimates,
            'threshold_used': threshold_tau
        }


# =============================================================================
# SECTION 3: FINANCIAL TIME-SERIES DATA GENERATION
# =============================================================================

def generate_synthetic_sp500(n_points: int = 500, seed: int = GLOBAL_SEED) -> np.ndarray:
    """
    Generate synthetic S&P 500-like log return series.
    Uses a regime-switching model with two volatility states
    to capture realistic market dynamics including crash episodes.

    Returns: 1D numpy array of log prices
    """
    local_rng = np.random.default_rng(seed)
    prices = np.zeros(n_points)
    prices[0] = np.log(3000)

    # Regime parameters
    mu_normal, sigma_normal = 0.0005, 0.01
    mu_crash, sigma_crash = -0.005, 0.04
    regime = 0  # 0 = normal, 1 = crash
    p_normal_to_crash = 0.005
    p_crash_to_normal = 0.05

    for t in range(1, n_points):
        if regime == 0:
            if local_rng.random() < p_normal_to_crash:
                regime = 1
            ret = local_rng.normal(mu_normal, sigma_normal)
        else:
            if local_rng.random() < p_crash_to_normal:
                regime = 0
            ret = local_rng.normal(mu_crash, sigma_crash)
        prices[t] = prices[t - 1] + ret

    return prices


def sliding_window_embedding(series: np.ndarray, window: int = 20,
                              stride: int = 5) -> np.ndarray:
    """
    Embed 1D time series into point cloud via sliding window.

    Window embedding: given x_1,...,x_T,
    Z_t = (x_t, x_{t+1}, ..., x_{t+w-1}) ∈ R^w

    Returns: array of shape (n_windows, window)
    Complexity: O(n_windows * window)
    """
    n = len(series)
    starts = range(0, n - window + 1, stride)
    return np.array([series[i:i + window] for i in starts])


# =============================================================================
# SECTION 4: BENCHMARKING SUITE (Phase 3 + Phase 4)
# =============================================================================

def benchmark_single_size(n_simplices_target: int, n_repetitions: int = 10,
                           k: int = 1, q_prec: int = 5) -> dict:
    """
    For a given target Laplacian dimension, benchmark classical vs quantum Betti estimation.

    We construct a controlled test: build a simplicial complex with known Betti numbers
    by generating a point cloud on a circle (β_0=1, β_1=1 for appropriate ε).

    Parameters:
    -----------
    n_simplices_target: approximate number of 1-simplices (controls Laplacian size)
    n_repetitions: number of QPE runs for statistics
    k: which Betti number (0 or 1)
    q_prec: QPE precision qubits

    Returns: dict with all measurements and statistics
    """
    # Generate point cloud with known topology
    # n points on a circle → β_0=1 (connected), β_1=1 (one loop) for appropriate ε
    n_pts = n_simplices_target  # n_pts directly controls Laplacian dim
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    # Add slight noise for realism
    local_rng = np.random.default_rng(GLOBAL_SEED + n_pts)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    # No noise: use exact circle for verified topology

    # Choose epsilon to capture circular structure
    # Neighbor spacing ~ 2π/n_pts, so epsilon ~ 1.5 * spacing
    spacing = 2 * np.sin(np.pi / n_pts)
    epsilon = 1.15 * spacing  # connects neighbors only, preserves circular topology

    # Classical Betti
    t_cls_start = time.perf_counter()
    vrc = VietorisRipsComplex(pts, epsilon, max_dim=2)
    lap = CombinorialLaplacian(vrc, k)
    beta_classical = lap.betti_number_classical()
    t_cls_end = time.perf_counter()
    classical_time = t_cls_end - t_cls_start

    n_k = lap.matrix_size()
    evals = lap.eigenvalues

    # Quantum estimates (n_repetitions independent runs)
    quantum_estimates = []
    quantum_raw = []
    t_qpe_start = time.perf_counter()

    for rep in range(n_repetitions):
        qpe = QuantumPhaseEstimation(
            lap.L, n_precision_qubits=q_prec,
            delta=0.5, shots=512,
            seed=GLOBAL_SEED + rep * 100 + n_pts
        )
        result = qpe.estimate_betti_threshold(n_samples=max(20, n_k * 2))
        quantum_estimates.append(result['betti_estimate'])
        quantum_raw.append(result['betti_raw'])

    t_qpe_end = time.perf_counter()
    quantum_time = (t_qpe_end - t_qpe_start) / n_repetitions

    q_arr = np.array(quantum_estimates)
    return {
        'n_pts': n_pts,
        'n_k_simplices': vrc.num_simplices(k),
        'laplacian_size': n_k,
        'epsilon': epsilon,
        'beta_classical': beta_classical,
        'eigenvalues': evals,
        'quantum_estimates': quantum_estimates,
        'quantum_mean': float(np.mean(q_arr)),
        'quantum_std': float(np.std(q_arr, ddof=1)) if len(q_arr) > 1 else 0.0,
        'quantum_abs_error': float(abs(np.mean(q_arr) - beta_classical)),
        'quantum_rel_error': float(abs(np.mean(q_arr) - beta_classical) / max(beta_classical, 1)),
        'classical_time': classical_time,
        'quantum_time_per_run': quantum_time,
        'n_repetitions': n_repetitions,
        'q_prec': q_prec
    }


def run_full_benchmark() -> list:
    """
    Run benchmarking across Laplacian sizes 4, 8, 16.
    Returns list of result dicts.
    """
    print("=" * 70)
    print("BENCHMARKING: Classical vs Quantum Betti Estimation")
    print("=" * 70)

    # Target Laplacian dimensions: 5, 8, 12 points on circle
    # Each gives exactly n_pts 1-simplices → Laplacian of size n_pts
    # All have β₁ = 1 (one loop) by construction
    targets = [5, 8, 12]

    results = []
    for lap_target in targets:
        print(f"\nBenchmarking n_pts={lap_target} circle (Laplacian dim = {lap_target})...")
        r = benchmark_single_size(
            n_simplices_target=lap_target,
            n_repetitions=10,
            k=1,
            q_prec=5
        )
        print(f"  Laplacian size: {r['laplacian_size']}")
        print(f"  Classical β₁: {r['beta_classical']}")
        print(f"  Quantum mean: {r['quantum_mean']:.3f} ± {r['quantum_std']:.3f}")
        print(f"  Abs error: {r['quantum_abs_error']:.3f}  Rel error: {r['quantum_rel_error']:.3f}")
        results.append(r)

    return results


def print_benchmark_table(results: list):
    """Print a formatted benchmark table."""
    print("\n" + "=" * 90)
    print(f"{'TABLE 1: Benchmarking Results — Quantum vs Classical Betti Number Estimation':^90}")
    print("=" * 90)
    header = (f"{'n_pts':>6} {'L_dim':>6} {'β₁_class':>9} "
              f"{'Q_mean':>8} {'Q_std':>8} {'Abs_err':>8} {'Rel_err':>8} "
              f"{'T_class(s)':>10} {'T_quant(s)':>10}")
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['n_pts']:>6} "
              f"{r['laplacian_size']:>6} "
              f"{r['beta_classical']:>9} "
              f"{r['quantum_mean']:>8.3f} "
              f"{r['quantum_std']:>8.3f} "
              f"{r['quantum_abs_error']:>8.3f} "
              f"{r['quantum_rel_error']:>8.3f} "
              f"{r['classical_time']:>10.4f} "
              f"{r['quantum_time_per_run']:>10.4f}")
    print("=" * 90)
    print("Note: Q_mean ± Q_std computed over 10 independent QPE runs.")
    print("Classical timing includes Rips complex construction + eigendecomposition.")
    print("Quantum timing is per-run (simulation overhead not counted as quantum cost).")


# =============================================================================
# SECTION 5: FINANCIAL ANALYSIS PIPELINE
# =============================================================================

def financial_tda_pipeline(n_series: int = 300, window: int = 15,
                            stride: int = 5, epsilon: float = 0.3,
                            k: int = 1) -> dict:
    """
    Full pipeline: synthetic S&P 500 → sliding window → TDA → crash detection.

    Returns dict with time series, windows, Betti curves, detection signal.
    """
    print("\nGenerating synthetic S&P 500 log price series...")
    series = generate_synthetic_sp500(n_series, seed=GLOBAL_SEED)
    windows = sliding_window_embedding(series, window, stride)
    n_w = len(windows)
    print(f"  {n_series} price points → {n_w} sliding windows of size {window}")

    print("Computing classical Betti numbers for each window...")
    betti_curves = {f'beta_{ki}': np.zeros(n_w) for ki in range(2)}
    matrix_sizes = np.zeros(n_w)
    timings = np.zeros(n_w)

    for t in range(n_w):
        if t % 20 == 0:
            print(f"  Window {t}/{n_w}...", end='\r')
        pts = windows[t].reshape(-1, 1)  # 1D embedding → points in R^1
        # Use 2D sliding window for richer topology
        if window >= 4:
            pts_2d = np.column_stack([windows[t][:-1], windows[t][1:]])
        else:
            pts_2d = pts

        t0 = time.perf_counter()
        vrc = VietorisRipsComplex(pts_2d, epsilon, max_dim=1)
        for ki in range(min(2, vrc.max_dim + 1)):
            lap = CombinorialLaplacian(vrc, ki)
            betti_curves[f'beta_{ki}'][t] = lap.betti_number_classical()
        timings[t] = time.perf_counter() - t0
        matrix_sizes[t] = vrc.num_simplices(1)

    print(f"\n  Done. Mean time per window: {timings.mean()*1000:.1f} ms")

    # Compute Lp-norm change signal
    beta1 = betti_curves['beta_1']
    diff = np.abs(np.diff(beta1))

    # Detect anomalies (spikes > 90th percentile)
    if diff.max() > 0:
        threshold_90 = np.percentile(diff, 90)
        crash_indices = np.where(diff > threshold_90)[0]
    else:
        threshold_90 = 0
        crash_indices = np.array([])

    return {
        'series': series,
        'windows': windows,
        'betti_curves': betti_curves,
        'diff_signal': diff,
        'threshold_90': threshold_90,
        'crash_indices': crash_indices,
        'timings': timings,
        'matrix_sizes': matrix_sizes,
        'epsilon': epsilon,
        'window_size': window
    }


# =============================================================================
# SECTION 6: COMPLEXITY ANALYSIS
# =============================================================================

def complexity_analysis():
    """
    Print and return the full complexity analysis.
    """
    analysis = {
        'classical': {
            'sliding_window': 'O(T * w)  — T time points, w window size',
            'rips_complex': 'O(n^{max_dim+1})  — exponential in dimension',
            'boundary_operator': 'O(n_k * n_{k+1})  — matrix filling',
            'eigendecomposition': 'O(n_k^3)  — BOTTLENECK via dense eigensolver',
            'total_per_window': 'O(n_k^3)  dominated by eigendecomposition',
            'total_all_windows': 'O(T/s * n_k^3)  — s = stride',
        },
        'quantum_theoretical': {
            'hamiltonian_simulation': 'O(s * polylog(n) * t / epsilon)  — sparse Ham. sim.',
            'qpe_precision': 'O(1/epsilon)  — linear in precision',
            'hilbert_space': 'O(log n_k)  — qubits scale logarithmically',
            'state_preparation': 'O(n_k)  — must prepare initial state',
            'total_qpe': 'O(poly(log n_k) / epsilon)  — THEORETICAL advantage',
            'nisq_constraints': 'Circuit depth O(n_k^2) in current simulation',
            'simulation_overhead': 'EXPONENTIAL in n_k (classical simulation of quantum)',
        },
        'bottleneck_note': (
            'The classical bottleneck is eigendecomposition O(n_k^3). '
            'Quantum QPE theoretically achieves O(poly(log n_k)) if:\n'
            '  (a) the Laplacian is sparse (s-sparse with s = O(polylog n_k))\n'
            '  (b) quantum RAM (QRAM) is available for state preparation\n'
            '  (c) fault-tolerant hardware is available\n'
            'NONE of these conditions hold in this SIMULATED implementation.'
        ),
        'honest_comparison': (
            'This project SIMULATES quantum computation classically. '
            'The simulation scales EXPONENTIALLY in the number of qubits, '
            'which is WORSE than the classical O(n_k^3) algorithm. '
            'The claimed quantum speedup is theoretical only.'
        )
    }
    return analysis


# =============================================================================
# SECTION 7: VISUALIZATION
# =============================================================================

def plot_all_results(benchmark_results: list, financial_results: dict,
                     save_prefix: str = '/home/claude/MSEF_project/') -> list:
    """Generate all publication-quality figures."""
    figures_saved = []
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                         'axes.titlesize': 12, 'axes.labelsize': 11})

    # -------------------------------------------------------------------------
    # Figure 1: Benchmark Results with Error Bars
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 1: Quantum vs Classical Betti Number Estimation',
                 fontsize=13, fontweight='bold')

    lap_sizes = [r['laplacian_size'] for r in benchmark_results]
    beta_true = [r['beta_classical'] for r in benchmark_results]
    q_means = [r['quantum_mean'] for r in benchmark_results]
    q_stds = [r['quantum_std'] for r in benchmark_results]
    abs_errs = [r['quantum_abs_error'] for r in benchmark_results]

    # Plot 1a: Estimates vs Ground Truth
    ax = axes[0]
    x = np.arange(len(benchmark_results))
    w = 0.35
    bars1 = ax.bar(x - w/2, beta_true, w, label='Classical (exact)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + w/2, q_means, w, yerr=q_stds, label='Quantum (estimated)',
                   color='coral', alpha=0.8, capsize=5)
    ax.set_xlabel('Laplacian Dimension')
    ax.set_ylabel('β₁ Estimate')
    ax.set_title('(a) Betti Number Estimates')
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={s}' for s in lap_sizes])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Plot 1b: Absolute Error
    ax = axes[1]
    colors = ['green' if e < 0.5 else 'orange' if e < 1.0 else 'red' for e in abs_errs]
    ax.bar(x, abs_errs, color=colors, alpha=0.8)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='0.5 threshold')
    ax.set_xlabel('Laplacian Dimension')
    ax.set_ylabel('|β₁_quantum - β₁_classical|')
    ax.set_title('(b) Absolute Estimation Error')
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={s}' for s in lap_sizes])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Plot 1c: Eigenvalue Spectrum of largest Laplacian
    ax = axes[2]
    r_last = benchmark_results[-1]
    evals_sorted = np.sort(r_last['eigenvalues'])
    ax.stem(range(len(evals_sorted)), evals_sorted,
            markerfmt='o', linefmt='C0-', basefmt='k-')
    # Use a reasonable threshold for display
    tau_display = evals_sorted[evals_sorted > 1e-10][0] * 0.3 if np.any(evals_sorted > 1e-10) else 0.1
    ax.axhline(tau_display,
               color='red', linestyle='--', alpha=0.7, label='QPE threshold τ (example)')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue λ')
    ax.set_title(f'(c) Laplacian Spectrum (n={r_last["laplacian_size"]})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname1 = save_prefix + 'figure1_benchmark.png'
    plt.savefig(fname1, dpi=150, bbox_inches='tight')
    plt.close()
    figures_saved.append(fname1)
    print(f"Saved: {fname1}")

    # -------------------------------------------------------------------------
    # Figure 2: Financial Pipeline Results
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle('Figure 2: Topological Analysis of Synthetic S&P 500 Returns',
                 fontsize=13, fontweight='bold')

    series = financial_results['series']
    betti_curves = financial_results['betti_curves']
    diff_sig = financial_results['diff_signal']
    crashes = financial_results['crash_indices']
    thresh = financial_results['threshold_90']

    ax = axes[0]
    ax.plot(series, color='navy', linewidth=1.2, label='Log Price')
    ax.set_title('(a) Synthetic S&P 500 Log Prices (Regime-Switching Model)')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Log Price')
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    colors_b = ['royalblue', 'crimson']
    for ki, col in enumerate(colors_b):
        bc = betti_curves.get(f'beta_{ki}', np.array([]))
        if len(bc) > 0:
            ax.plot(bc, color=col, linewidth=1.5, label=f'β_{ki}', alpha=0.9)
    ax.set_title('(b) Betti Numbers β₀, β₁ Over Sliding Windows')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Betti Number')
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(diff_sig, color='darkred', linewidth=1.5, label='|Δβ₁| change signal')
    ax.axhline(thresh, color='orange', linestyle='--', linewidth=1.5,
               label=f'90th-percentile threshold = {thresh:.3f}')
    if len(crashes) > 0:
        ax.scatter(crashes, diff_sig[crashes], color='red', s=60,
                   zorder=5, marker='X', label=f'Detected anomalies ({len(crashes)})')
    ax.set_title('(c) Topological Change Signal & Anomaly Detection')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('L¹ Betti Difference')
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fname2 = save_prefix + 'figure2_financial.png'
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    plt.close()
    figures_saved.append(fname2)
    print(f"Saved: {fname2}")

    # -------------------------------------------------------------------------
    # Figure 3: QPE Phase Distribution (diagnostic)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 3: QPE Phase Distribution & Eigenvalue Estimation',
                 fontsize=13, fontweight='bold')

    # Use the medium-size Laplacian result
    mid = benchmark_results[1]
    # Re-run one QPE to get phase data
    pts_mid = mid['n_pts']
    theta_mid = np.linspace(0, 2 * np.pi, pts_mid, endpoint=False)
    pts_arr = np.column_stack([np.cos(theta_mid), np.sin(theta_mid)])
    pts_arr += np.random.default_rng(GLOBAL_SEED + pts_mid).normal(0, 0.05, pts_arr.shape)
    eps_mid = 1.5 * (2 * np.pi / pts_mid)
    vrc_mid = VietorisRipsComplex(pts_arr, eps_mid, max_dim=2)
    lap_mid = CombinorialLaplacian(vrc_mid, 1)

    qpe_diag = QuantumPhaseEstimation(lap_mid.L, n_precision_qubits=5,
                                       delta=0.5, seed=GLOBAL_SEED)
    diag_result = qpe_diag.estimate_betti_threshold(n_samples=50)

    ax = axes[0]
    if len(diag_result['phases']) > 0:
        ax.hist(diag_result['phases'], bins=20, color='steelblue', alpha=0.7,
                edgecolor='white')
        ax.axvline(diag_result['threshold_used'] / (2 * np.pi * qpe_diag.lambda_max /
                   qpe_diag.delta), color='red', linestyle='--',
                   label=f'τ phase threshold')
        ax.set_xlabel('Measured QPE Phase φ')
        ax.set_ylabel('Count')
        ax.set_title('(a) Distribution of QPE Phase Measurements')
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1]
    if len(diag_result['lambda_estimates']) > 0:
        ax.hist(diag_result['lambda_estimates'], bins=20, color='coral', alpha=0.7,
                edgecolor='white')
        ax.axvline(diag_result['threshold_used'], color='red', linestyle='--',
                   label=f'Threshold τ = {diag_result["threshold_used"]:.3f}')
        # Mark true eigenvalues
        for ev in np.sort(lap_mid.eigenvalues):
            ax.axvline(ev, color='navy', linestyle=':', alpha=0.5)
        ax.set_xlabel('Eigenvalue Estimate λ̂')
        ax.set_ylabel('Count')
        ax.set_title('(b) Quantum Eigenvalue Estimates vs True Spectrum')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.text(0.55, 0.92, f'True β₁ = {lap_mid.betti_number_classical()}\n'
                             f'QPE β₁ = {diag_result["betti_estimate"]}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    fname3 = save_prefix + 'figure3_qpe_diagnostic.png'
    plt.savefig(fname3, dpi=150, bbox_inches='tight')
    plt.close()
    figures_saved.append(fname3)
    print(f"Saved: {fname3}")

    # -------------------------------------------------------------------------
    # Figure 4: Complexity Scaling
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 4: Computational Complexity Comparison', fontsize=13, fontweight='bold')

    n_range = np.logspace(1, 3, 100)

    ax = axes[0]
    ax.loglog(n_range, n_range ** 3, 'b-', linewidth=2, label='Classical O(n³)')
    ax.loglog(n_range, n_range ** 2, 'g--', linewidth=2, label='Simulation O(n²·sim)')
    ax.loglog(n_range, np.log2(n_range) ** 2 * 10, 'r-.', linewidth=2,
              label='Theoretical Quantum O(log²n)')
    ax.set_xlabel('Laplacian Dimension n')
    ax.set_ylabel('Operations (log scale)')
    ax.set_title('(a) Theoretical Complexity Scaling')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    lap_sizes_bench = [r['laplacian_size'] for r in benchmark_results]
    class_times = [r['classical_time'] for r in benchmark_results]
    quant_times = [r['quantum_time_per_run'] for r in benchmark_results]
    ax.plot(lap_sizes_bench, class_times, 'bo-', linewidth=2,
            markersize=8, label='Classical (measured)')
    ax.plot(lap_sizes_bench, quant_times, 'rs--', linewidth=2,
            markersize=8, label='QPE simulation (measured)')
    ax.set_xlabel('Laplacian Dimension n')
    ax.set_ylabel('Wall-Clock Time (seconds)')
    ax.set_title('(b) Measured Runtime (SIMULATION)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.85, 'Note: QPE simulation runs SLOWER\nthan classical on current hardware.',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    fname4 = save_prefix + 'figure4_complexity.png'
    plt.savefig(fname4, dpi=150, bbox_inches='tight')
    plt.close()
    figures_saved.append(fname4)
    print(f"Saved: {fname4}")

    return figures_saved


# =============================================================================
# SECTION 8: CLAIM VALIDATION
# =============================================================================

def validate_title():
    return {
        'original_title': (
            'Quantum Computation of Topological Invariants '
            'for Financial Time-Series Analysis'
        ),
        'verdict': 'PARTIALLY JUSTIFIED — with necessary qualification',
        'justification': (
            'The title is mathematically legitimate IF properly scoped:\n'
            '(+) Topological invariants (Betti numbers) ARE computed.\n'
            '(+) Financial time-series data IS analyzed.\n'
            '(+) A quantum algorithm (QPE) IS used for the computation.\n'
            '(-) The quantum computation is SIMULATED classically, not run on\n'
            '    actual quantum hardware.\n'
            '(-) No quantum speedup is demonstrated — the simulation is slower.\n'
            '(-) The original QPE method (p(0)*2^q) was mathematically incorrect.\n'
        ),
        'corrected_title': (
            'Simulated Quantum Phase Estimation for Betti Number Computation '
            'in Financial Time-Series Topological Analysis'
        ),
        'corrected_title_rationale': (
            'The word "Simulated" is essential for academic honesty. '
            '"Phase Estimation for Betti Number Computation" is more precise '
            'than "Computation of Topological Invariants" since we specifically '
            'use QPE on the Laplacian spectrum, and Betti numbers are the specific '
            'invariant being estimated.'
        )
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import os
    os.makedirs('/home/claude/MSEF_project', exist_ok=True)

    print("\n" + "=" * 70)
    print("MSEF 2026: Quantum TDA for Financial Time-Series Analysis")
    print("Complete Implementation & Audit")
    print("=" * 70)

    # Phase 1: Audit
    print("\n[PHASE 1] AUDIT SUMMARY")
    print("-" * 50)
    print("✗ Original QPE: β = p(0)*2^q — INCORRECT (fixed)")
    print("✗ No classical baseline comparison — FIXED")
    print("✗ No benchmarking with statistics — FIXED")
    print("✗ No complexity analysis — FIXED")
    print("✗ Limitations not stated — FIXED")
    print("✓ Topological invariants are computed (Betti numbers via Laplacian)")

    # Phase 3+4: Benchmarking
    print("\n[PHASE 3+4] RUNNING BENCHMARKS...")
    bench_results = run_full_benchmark()
    print_benchmark_table(bench_results)

    # Phase 5: Complexity
    print("\n[PHASE 5] COMPLEXITY ANALYSIS")
    comp = complexity_analysis()
    print("\nClassical bottleneck:", comp['classical']['eigendecomposition'])
    print("Quantum theoretical:", comp['quantum_theoretical']['total_qpe'])
    print("\nHonest assessment:", comp['honest_comparison'])

    # Financial pipeline
    print("\n[FINANCIAL PIPELINE]")
    fin_results = financial_tda_pipeline(n_series=300, window=12, stride=4, epsilon=0.3)
    print(f"  Detected {len(fin_results['crash_indices'])} anomalies in {len(fin_results['windows'])} windows")

    # Figures
    print("\n[GENERATING FIGURES]")
    figs = plot_all_results(bench_results, fin_results)
    print(f"  Generated {len(figs)} figures")

    # Phase 9: Claim validation
    print("\n[PHASE 9] CLAIM VALIDATION")
    cv = validate_title()
    print(f"  Verdict: {cv['verdict']}")
    print(f"  Corrected title: {cv['corrected_title']}")

    # Save results for paper generation
    results_summary = {
        'benchmark': [
            {k: v for k, v in r.items()
             if k not in ('eigenvalues', 'quantum_estimates')}
            for r in bench_results
        ],
        'financial': {
            'n_windows': len(fin_results['windows']),
            'n_crashes': int(len(fin_results['crash_indices'])),
            'threshold_90': float(fin_results['threshold_90']),
            'mean_beta1': float(fin_results['betti_curves']['beta_1'].mean()),
            'std_beta1': float(fin_results['betti_curves']['beta_1'].std()),
        },
        'claim_validation': cv,
        'figures': figs
    }

    with open('/home/claude/MSEF_project/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n✓ All computations complete. Results saved.")
    print("✓ Proceeding to paper/notebook generation.")
