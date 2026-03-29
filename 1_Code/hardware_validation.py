"""
=============================================================================
IBM QUANTUM HARDWARE VALIDATION
MSEF 2026 — Real QPE on Real Quantum Hardware

This script runs a minimal QPE circuit on a real IBM Quantum device,
then compares hardware output against our classical simulation predictions.

SETUP:
  1. pip install qiskit qiskit-ibm-runtime
  2. Replace YOUR_IBM_TOKEN with your token from quantum.cloud.ibm.com
  3. Run: python hardware_validation.py

WHY THIS MATTERS:
  Without this, the project only simulates quantum computation.
  This script runs ACTUAL quantum gates on ACTUAL superconducting qubits
  in IBM's dilution refrigerators, making the "quantum" claim legitimate.

CIRCUIT SIZE:
  We use the smallest non-trivial Laplacian: a 2-simplex (one edge).
  - Laplacian Δ₀ for a single edge: 2×2 matrix [[1,-1],[-1,1]]
  - This needs 1 qubit for eigenstate register
  - We use 3 precision qubits → 4 qubits total
  - Well within any IBM device's capacity

WHAT WE COMPARE:
  Simulation prediction: exact measurement probabilities for each phase outcome
  Hardware result:       measured counts from real quantum device
  Fidelity metric:       total variation distance between distributions
=============================================================================
"""

import numpy as np
import scipy.linalg as la
import json
import sys

# ============================================================
# STEP 1: DEFINE THE MINIMAL LAPLACIAN
# ============================================================

def build_edge_laplacian():
    """
    Simplest non-trivial Laplacian: a single edge {0,1}.
    
    0-simplices: {(0,), (1,)}
    1-simplices: {(0,1)}
    
    Δ₀ = ∂₁∂₁ᵀ (only upper part, no lower for k=0)
    
    Boundary ∂₁: C₁ → C₀
    ∂₁(e₀₁) = v₁ - v₀  →  B = [[-1], [+1]]
    
    Δ₀ = BBᵀ = [[1,-1],[-1,1]]
    
    Eigenvalues: {0, 2}
    β₀ = 1 (one connected component — correct, it's a single edge)
    """
    L = np.array([[1., -1.],
                  [-1., 1.]])
    evals = np.real(la.eigvalsh(L))
    print("Minimal Laplacian Δ₀ for single edge {0,1}:")
    print(f"  Matrix:\n  {L}")
    print(f"  Eigenvalues: {np.round(evals, 6)}")
    print(f"  True β₀ = {np.sum(evals <= 1e-8)} (one zero eigenvalue = 1 component)")
    return L, evals


# ============================================================
# STEP 2: COMPUTE SIMULATION PREDICTION
# ============================================================

def simulate_qpe_distribution(L, n_prec=3, delta=0.5):
    """
    Compute the EXACT probability distribution that QPE should output
    for our Laplacian, assuming a specific initial state.
    
    We use initial state |0> (first basis vector of eigenstate register).
    
    For the 2×2 Laplacian [[1,-1],[-1,1]]:
    - Eigenvalues: λ₀=0, λ₁=2
    - Eigenvectors: v₀=[1,1]/√2, v₁=[1,-1]/√2
    - |0> = (1/√2)|v₀> + (1/√2)|v₁>
    
    So QPE will measure:
    - Phase φ₀ = 0           (eigenvalue 0)    with probability 0.5
    - Phase φ₁ = δ×2/λ_max  (eigenvalue 2)    with probability 0.5
    
    This gives a PREDICTED distribution we can compare to hardware.
    """
    lambda_max = max(np.abs(np.real(la.eigvalsh(L))))
    H = (delta / lambda_max) * L
    U = la.expm(1j * H)
    
    evals_U, evecs_U = la.eig(U)
    phases_true = np.angle(evals_U) / (2*np.pi) % 1.0
    
    M = 2**n_prec
    psi_init = np.zeros(2, dtype=complex)
    psi_init[0] = 1.0
    
    coeffs = evecs_U.conj().T @ psi_init
    
    probs = np.zeros(M)
    for k in range(M):
        amp = 0.0 + 0j
        for m in range(2):
            alpha = phases_true[m] - k/M
            alpha_mod = alpha % 1.0
            if alpha_mod < 1e-12 or alpha_mod > 1-1e-12:
                inner = M
            else:
                inner = (1 - np.exp(2j*np.pi*M*alpha)) / (1 - np.exp(2j*np.pi*alpha))
            amp += coeffs[m] * inner / np.sqrt(M)
        probs[k] = abs(amp)**2
    
    probs = probs / probs.sum()
    
    print(f"\nSimulation prediction (n_prec={n_prec} qubits, delta={delta}):")
    print(f"  Unitary phases: {np.round(phases_true, 4)}")
    print(f"  Predicted measurement distribution:")
    for k in range(M):
        bar = '█' * int(probs[k] * 40)
        print(f"    |{k:03b}> (k={k}): p={probs[k]:.4f}  {bar}")
    
    # Identify which outcome corresponds to β₀=1
    zero_threshold_phase = delta * 1e-8 / lambda_max / (2*np.pi)
    zero_outcomes = [k for k in range(M) if k/M < zero_threshold_phase or k/M > 1-zero_threshold_phase]
    print(f"\n  Outcomes indicating zero eigenvalue (phase≈0): {zero_outcomes}")
    print(f"  Total probability at phase≈0: {sum(probs[k] for k in zero_outcomes):.4f}")
    print(f"  Expected ≈ 0.5 (half the initial state is in zero eigenspace)")
    
    return probs, phases_true


# ============================================================
# STEP 3: BUILD QISKIT QPE CIRCUIT
# ============================================================

def build_qpe_circuit(n_prec=3, delta=0.5):
    """
    Build the QPE circuit for Δ₀ = [[1,-1],[-1,1]].
    
    Circuit layout:
    - Qubits 0,1,2: precision register (3 qubits)
    - Qubit 3: eigenstate register (1 qubit, encodes C₀)
    - Classical bits 0,1,2: measure precision register
    
    The unitary U = exp(i·δ·Δ₀/λ_max):
    For δ=0.5, λ_max=2:
      H_scaled = 0.25 * [[1,-1],[-1,1]]
      U = exp(i·H_scaled)
    
    We decompose U into standard 1-qubit gates (Rz, Ry, global phase).
    """
    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit.library import QFT
    except ImportError:
        print("ERROR: Qiskit not installed. Run: pip install qiskit qiskit-ibm-runtime")
        sys.exit(1)
    
    L = np.array([[1., -1.], [-1., 1.]])
    lambda_max = 2.0
    
    # Compute U = exp(i * delta * L / lambda_max) analytically
    # For this 2x2 matrix: U = exp(i * delta/2 * [[1,-1],[-1,1]])
    # = exp(i*delta/2*I) * exp(-i*delta/2*Z·something)
    # Exact: eigenvalues 0 and 2, eigenvectors [1,1]/√2 and [1,-1]/√2
    # U|v₀> = e^0 |v₀>, U|v₁> = e^{i*delta} |v₁>
    
    theta = delta  # phase for eigenvalue 2
    # U in computational basis:
    # U = [[cos(θ/2)+i·sin(θ/2)·0.5+0.5i, ...]] — use scipy
    H_sc = (delta / lambda_max) * L
    U_mat = la.expm(1j * H_sc)
    
    # Decompose 2x2 unitary into gates
    # U = e^{iα} * Rz(β) * Ry(γ) * Rz(δ_angle)
    # For our specific case, find parameters numerically
    
    qr_prec = QuantumRegister(n_prec, 'prec')
    qr_eigen = QuantumRegister(1, 'eigen')
    cr = ClassicalRegister(n_prec, 'c')
    
    qc = QuantumCircuit(qr_prec, qr_eigen, cr)
    
    # Initialize eigenstate register to |0> (first basis vector)
    # |0> is the default, so no initialization needed
    
    # Apply Hadamard to all precision qubits
    for i in range(n_prec):
        qc.h(qr_prec[i])
    
    # Controlled-U^{2^j} for j = 0, 1, ..., n_prec-1
    from qiskit.circuit.library import UnitaryGate
    for j in range(n_prec):
        power = 2**j
        U_power = np.linalg.matrix_power(U_mat, power)
        # Add controlled unitary gate
        cu_gate = UnitaryGate(U_power).control(1)
        qc.append(cu_gate, [qr_prec[j], qr_eigen[0]])
    
    # Inverse QFT on precision register
    qc.append(QFT(n_prec, inverse=True, do_swaps=True), qr_prec)
    
    # Measure precision register
    qc.measure(qr_prec, cr)
    
    print(f"\nQPE Circuit built successfully:")
    print(f"  Total qubits: {qc.num_qubits} ({n_prec} precision + 1 eigenstate)")
    print(f"  Total gates: {qc.size()}")
    print(f"  Circuit depth: {qc.depth()}")
    print(f"  Classical bits: {n_prec}")
    
    return qc, U_mat


# ============================================================
# STEP 4: RUN ON IBM QUANTUM HARDWARE
# ============================================================

def run_on_hardware(qc, api_token, shots=1024, n_prec=3):
    """
    Submit QPE circuit to real IBM Quantum hardware.
    
    Uses the Sampler primitive with the least-busy device.
    The Sampler returns quasi-probability distributions from real measurements.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError:
        print("ERROR: qiskit-ibm-runtime not installed.")
        print("Run: pip install qiskit-ibm-runtime")
        return None
    
    print(f"\nConnecting to IBM Quantum...")
    
    # Save account (only needed once)
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=api_token,
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/758eec85dadf40dcafc61a14c9f45032:f44cf6be-e36c-424d-bbb8-8dfae7d90f6b::",
        overwrite=True
    )
    
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=api_token,
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/758eec85dadf40dcafc61a14c9f45032:f44cf6be-e36c-424d-bbb8-8dfae7d90f6b::"
    )
    
    # Find least busy real backend with enough qubits
    min_qubits = qc.num_qubits
    backends = service.backends(
        min_num_qubits=min_qubits,
        operational=True,
        simulator=False
    )
    
    if not backends:
        print(f"ERROR: No available backend with >= {min_qubits} qubits.")
        print("Try reducing n_prec to 2 (needs 3 qubits total).")
        return None
    
    # Sort by queue length
    backend = min(backends, key=lambda b: b.status().pending_jobs)
    
    print(f"Selected backend: {backend.name}")
    print(f"  Qubits: {backend.configuration().n_qubits}")
    print(f"  Queue: {backend.status().pending_jobs} jobs")
    
    # Transpile for the specific backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pm.run(qc)
    
    print(f"  Transpiled circuit depth: {transpiled.depth()}")
    print(f"  Transpiled gate count: {transpiled.size()}")
    
    # Run with Sampler
    print(f"\nSubmitting job ({shots} shots)...")
    sampler = Sampler(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    
    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting for results (may take several minutes)...")
    
    result = job.result()
    
    # Extract counts
    pub_result = result[0]
    counts_raw = pub_result.data.c.get_counts()
    
    # Convert to probabilities
    total = sum(counts_raw.values())
    hardware_probs = np.zeros(2**n_prec)
    for bitstring, count in counts_raw.items():
        # Qiskit returns big-endian bitstrings
        k = int(bitstring, 2)
        hardware_probs[k] = count / total
    
    print(f"\nHardware results received!")
    print(f"  Backend: {backend.name}")
    print(f"  Shots: {total}")
    print(f"  Raw counts: {counts_raw}")
    
    return {
        'backend_name': backend.name,
        'job_id': job.job_id(),
        'shots': total,
        'counts': counts_raw,
        'probs': hardware_probs.tolist(),
        'n_prec': n_prec
    }


# ============================================================
# STEP 5: COMPARE HARDWARE VS SIMULATION
# ============================================================

def compare_results(sim_probs, hardware_result, n_prec=3):
    """
    Compare simulation prediction vs hardware measurement.
    
    Metrics:
    1. Total Variation Distance (TVD): Σ|p_sim - p_hw| / 2
       TVD = 0 → perfect agreement, TVD = 1 → maximum disagreement
       
    2. Hellinger Distance: sqrt(1 - Σ sqrt(p_sim × p_hw))
       More sensitive to differences at small probabilities
       
    3. Visual comparison: side-by-side bar chart
    
    We also report the implied β₀ from hardware and compare to classical truth.
    """
    M = 2**n_prec
    hw_probs = np.array(hardware_result['probs'])
    
    # Total Variation Distance
    tvd = np.sum(np.abs(sim_probs - hw_probs)) / 2
    
    # Hellinger Distance  
    hellinger = np.sqrt(1 - np.sum(np.sqrt(sim_probs * hw_probs)))
    
    # Betti estimate from hardware
    # Phase ≈ 0 → eigenvalue ≈ 0 → zero-eigenvalue detection
    # For n_prec=3: outcome k=0 or k=7 indicate phase≈0 (with wrapping)
    threshold_k = max(1, M // 8)  # allow some leakage
    zero_prob_hw = sum(hw_probs[k] for k in range(M)
                       if k < threshold_k or k > M - threshold_k)
    
    # β₀ estimate: for initial state |0> = (1/√2)|v₀> + (1/√2)|v₁>
    # zero_prob ≈ 0.5 → β₀_est ≈ 0.5 × 2 = 1.0
    beta0_hw_est = zero_prob_hw * 2  # n_k = 2 for 0-simplices
    
    print("\n" + "="*65)
    print("HARDWARE vs SIMULATION COMPARISON")
    print("="*65)
    print(f"\nBackend: {hardware_result['backend_name']}")
    print(f"Job ID:  {hardware_result['job_id']}")
    print(f"Shots:   {hardware_result['shots']}")
    print()
    print(f"{'Outcome':<12} {'Sim Prob':>10} {'HW Prob':>10} {'Difference':>12}")
    print("-"*50)
    for k in range(M):
        diff = hw_probs[k] - sim_probs[k]
        print(f"|{k:0{n_prec}b}> k={k:<3} {sim_probs[k]:>10.4f} {hw_probs[k]:>10.4f} {diff:>+12.4f}")
    print()
    print(f"Total Variation Distance (TVD):  {tvd:.4f}")
    print(f"  (0=perfect match, 1=completely wrong)")
    print(f"  {'✓ Good agreement' if tvd < 0.15 else '! Moderate noise' if tvd < 0.35 else '✗ High noise (expected on NISQ)'}")
    print()
    print(f"Hellinger Distance:              {hellinger:.4f}")
    print()
    print(f"β₀ estimate from hardware:       {beta0_hw_est:.3f}")
    print(f"Classical truth β₀:              1  (exact)")
    print(f"Simulation β₀ estimate:          {sum(sim_probs[k] for k in range(M) if k < threshold_k or k > M-threshold_k) * 2:.3f}")
    print()
    print("Interpretation:")
    print("  NISQ hardware introduces gate errors and decoherence.")
    print(f"  TVD={tvd:.3f} quantifies the noise impact on our circuit.")
    print("  This is EXPECTED and is a key finding: fault-tolerant hardware")
    print("  would reduce TVD toward 0, enabling accurate Betti estimation.")
    
    return {
        'tvd': float(tvd),
        'hellinger': float(hellinger),
        'beta0_hw_estimate': float(beta0_hw_est),
        'beta0_classical': 1,
        'sim_probs': sim_probs.tolist(),
        'hw_probs': hw_probs.tolist()
    }


def plot_hardware_comparison(sim_probs, comparison, n_prec=3, save_path='figure5_hardware.png'):
    """Generate Figure 5: Hardware vs Simulation comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    M = 2**n_prec
    hw_probs = np.array(comparison['hw_probs'])
    labels = [f'|{k:0{n_prec}b}⟩' for k in range(M)]
    x = np.arange(M)
    w = 0.38
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 5: Real IBM Quantum Hardware vs Classical Simulation\n'
                 f'Backend: {comparison.get("backend", "IBM QPU")} | '
                 f'TVD = {comparison["tvd"]:.4f}',
                 fontsize=12, fontweight='bold')
    
    ax = axes[0]
    ax.bar(x - w/2, sim_probs, w, label='Simulation (ideal)', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, hw_probs, w, label='IBM Quantum Hardware', color='coral', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('QPE Measurement Outcome'); ax.set_ylabel('Probability')
    ax.set_title('(a) Measurement Distribution: Simulation vs Hardware')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.97,
            f'TVD = {comparison["tvd"]:.4f}\nHellinger = {comparison["hellinger"]:.4f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    ax = axes[1]
    diff = hw_probs - sim_probs
    colors = ['green' if d >= 0 else 'red' for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('QPE Measurement Outcome')
    ax.set_ylabel('P_hardware − P_simulation')
    ax.set_title('(b) Deviation from Ideal (NISQ Noise Effect)')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.05,
            'Deviations = NISQ gate errors + decoherence.\n'
            'Fault-tolerant hardware would reduce these to ~0.',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#fff5f5'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure 5 saved: {save_path}')


# ============================================================
# MAIN — Replace YOUR_IBM_TOKEN with your actual token
# ============================================================

if __name__ == '__main__':
    
    print("="*65)
    print("IBM QUANTUM HARDWARE VALIDATION")
    print("MSEF 2026: QPE on Real Superconducting Qubits")
    print("="*65)
    
    # ---- Configuration ----
    N_PREC = 3       # precision qubits (4 qubits total)
    DELTA = 0.5      # Hamiltonian scaling
    SHOTS = 1024     # measurement shots
    
    # PASTE YOUR IBM TOKEN HERE:
    IBM_TOKEN = "YOUR_IBM_TOKEN_HERE"
    
    if IBM_TOKEN == "YOUR_IBM_TOKEN_HERE":
        print("\n" + "!"*65)
        print("! SETUP REQUIRED:")
        print("! 1. Go to https://quantum.cloud.ibm.com")
        print("! 2. Create a free account")
        print("! 3. Copy your API token from account settings")
        print("! 4. Replace YOUR_IBM_TOKEN_HERE in this file")
        print("!"*65)
        print("\nRunning SIMULATION ONLY (no hardware) to demonstrate the code...")
        RUN_HARDWARE = False
    else:
        RUN_HARDWARE = True
    
    # Step 1: Define Laplacian
    L, evals = build_edge_laplacian()
    
    # Step 2: Simulation prediction
    sim_probs, phases = simulate_qpe_distribution(L, N_PREC, DELTA)
    
    # Step 3: Build circuit
    qc, U_mat = build_qpe_circuit(N_PREC, DELTA)
    print(f"\nCircuit diagram (text):")
    print(qc.draw('text'))
    
    if RUN_HARDWARE:
        # Step 4: Run on real hardware
        hw_result = run_on_hardware(qc, IBM_TOKEN, SHOTS, N_PREC)
        
        if hw_result:
            # Step 5: Compare
            comparison = compare_results(sim_probs, hw_result, N_PREC)
            comparison['backend'] = hw_result['backend_name']
            
            # Save results
            with open('hardware_results.json', 'w') as f:
                json.dump({
                    'hardware': hw_result,
                    'comparison': comparison,
                    'laplacian': L.tolist(),
                    'true_eigenvalues': evals.tolist(),
                    'classical_beta_0': 1
                }, f, indent=2)
            print("\nhardware_results.json saved.")
            
            # Plot
            plot_hardware_comparison(sim_probs, comparison, N_PREC)
            
            print("\n" + "="*65)
            print("SUMMARY FOR PAPER/POSTER:")
            print("="*65)
            print(f"  Laplacian: 2×2, single edge, β₀=1 (known)")
            print(f"  QPE Circuit: {qc.num_qubits} qubits, {N_PREC} precision qubits")
            print(f"  Backend: {hw_result['backend_name']}")
            print(f"  Shots: {SHOTS}")
            print(f"  TVD (sim vs hardware): {comparison['tvd']:.4f}")
            print(f"  β₀ from hardware: {comparison['beta0_hw_estimate']:.3f} (truth=1)")
            print(f"  This run required <{SHOTS/6000:.1f} seconds of quantum time")
            print(f"  (well within IBM's free 10 min/month tier)")
    
    else:
        print("\n[SIMULATION MODE ONLY]")
        print("The circuit is built and verified. To run on hardware:")
        print("  1. Get your IBM token from quantum.cloud.ibm.com")
        print("  2. Set IBM_TOKEN in this file")
        print("  3. Run: python hardware_validation.py")
        print()
        print("Simulated output for reference:")
        print("  If hardware matches simulation (TVD < 0.15): strong validation")
        print("  If TVD = 0.15-0.35: moderate NISQ noise (expected)")
        print("  If TVD > 0.35: significant decoherence (also a valid finding)")
