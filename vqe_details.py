import torch
import cudaq
import numpy as np
from scipy.optimize import minimize
from tqdm import trange
from cudaq import SpinOperator
from vmc_cal import efficient_parallel_sampler, local_energy_batch

@cudaq.kernel
def prepare_only(initial_state: list[complex]):
    _ = cudaq.qvector(initial_state)  

@cudaq.kernel
def vqe_ansatz_kernel(thetas: list[float], initial_state: list[complex], n_layers: int, n_qubits: int):
    """
    Hardware-efficient ansatz for VQE.
    The kernel applies a series of single-qubit rotations and entangling gates.
    """
    # Initialize the qubit register directly from the passed-in numpy array.
    qubits = cudaq.qvector(initial_state)

    # Apply the hardware-efficient ansatz.
    theta_idx = 0
    for _ in range(n_layers):
        # Single-qubit rotation: RY then RZ
        for q in range(n_qubits):
            ry(thetas[theta_idx], qubits[q])
            theta_idx += 1
            rz(thetas[theta_idx], qubits[q])
            theta_idx += 1

        # Ring CNOTs  
        for q in range(n_qubits - 1):
            z.ctrl(qubits[q], qubits[q + 1])

    # Optional: final RY-RZ layer after all entangling blocks
    for q in range(n_qubits):
        ry(thetas[theta_idx], qubits[q])
        theta_idx += 1
        rz(thetas[theta_idx], qubits[q])
        theta_idx += 1

def build_full_state_from_samples(nqs_model, samples, n_qubits, device="cpu"):
    """
    Build a full statevector from a set of sampled bitstrings.
    Only amplitudes of sampled configurations are filled (others = 0).
    """
    dev = next(nqs_model.parameters()).device
    with torch.no_grad():
        unique_samples, _ = torch.unique(samples, dim=0, return_inverse=True)
        
        sigmas = unique_samples.to(dev).float()
        log_psi = nqs_model.log_prob(sigmas)   
        amps = torch.exp(log_psi).to(torch.complex64)

        bits = (1.0 - unique_samples.to(torch.float32)) / 2.0
        powers_of_two = 2 ** torch.arange(0, n_qubits, device=bits.device, dtype=bits.dtype)
        int_indices = (bits @ powers_of_two).long()

        full_state_vector = torch.zeros(2**n_qubits, dtype=torch.complex64, device=dev)
        full_state_vector[int_indices] = amps

        psi = full_state_vector / torch.linalg.norm(full_state_vector)

        return np.ascontiguousarray(psi.detach().cpu().numpy().astype(np.complex64))

def run_vqe_fine_tuning(molecule_ham, nqs_model, n_qubits, n_electrons,
                        n_layers, vmc_params, qham_of, max_vqe_iterations, device='cpu'):
    """
    Hybrid VQE optimization:
    1. Adam (warm-up, stochastic gradient descent style)
    2. L-BFGS-B with parameter-shift gradients (fine-tuning)
    """

    molecule_ham = SpinOperator(qham_of)

    # -----------------------------
    # Step 1) Build initial state
    # -----------------------------
    print(f"  [VQE Step] Sampling {vmc_params['n_samples']} configurations from NQS...")
    samples = efficient_parallel_sampler(
        nqs_model,
        vmc_params['n_samples'] // vmc_params['n_chains'],
        vmc_params['n_chains'],
        n_qubits,
        vmc_params['burn_in_steps'],
        vmc_params['step_intervals'],
        device
    )

    initial_state_np = build_full_state_from_samples(nqs_model, samples, n_qubits, device)
    print(f"  [VQE Step] Initial state built from {samples.shape[0]} samples.")

    E0 = cudaq.observe(prepare_only, molecule_ham, initial_state_np).expectation()
    print(f"[Init] Energy of NQS (CUDA-Q) = {E0:.8f} Ha")

    # -----------------------------
    # Step 2) Define cost & grad
    # -----------------------------
    parameter_count = 2 * n_qubits * n_layers + 2 * n_qubits
    print(f"[VQE] parameter_count = {parameter_count}")

    def cost(theta: np.ndarray) -> float:
        energy = cudaq.observe(
            vqe_ansatz_kernel, molecule_ham,
            theta, initial_state_np, n_layers, n_qubits
        ).expectation()
        print(f"[DEBUG] Energy: {energy:.6f}")
        return energy

    shift = np.pi / 2.0
    half = 0.5

    def grad(theta: np.ndarray) -> np.ndarray:
        g = np.zeros_like(theta)
        for i in range(theta.size):
            t_plus = theta.copy();  t_plus[i] += shift
            t_minus = theta.copy(); t_minus[i] -= shift
            Ep = cudaq.observe(vqe_ansatz_kernel, molecule_ham, t_plus, initial_state_np, n_layers, n_qubits).expectation()
            Em = cudaq.observe(vqe_ansatz_kernel, molecule_ham, t_minus, initial_state_np, n_layers, n_qubits).expectation()
            g[i] = half * (Ep - Em)
        return g

    # -----------------------------
    # Step 3) Random init (avoid barren plateau at θ=0)
    # -----------------------------
    rng = np.random.default_rng(42)
    theta0 = rng.normal(0, 0.1, parameter_count)
    print(f"[Check] cost(theta0) = {cost(theta0):.8f} Ha")

    # -----------------------------
    # Step 4) Warm-up with Adam
    # -----------------------------
    print("  [VQE Step] Warm-up with Adam...")
    lr = 0.01
    adam_iters = max_vqe_iterations // 3  # use ~1/3 iterations for Adam
    m = np.zeros_like(theta0)
    v = np.zeros_like(theta0)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    theta = theta0.copy()
    for t in range(1, adam_iters + 1):
        g = grad(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        E = cost(theta)
        print(f"  [Adam {t}/{adam_iters}] Energy = {E:.6f} Ha")

    # -----------------------------
    # Step 5) Fine-tuning with L-BFGS-B
    # -----------------------------
    print("  [VQE Step] Fine-tuning with L-BFGS-B...")
    result = minimize(
        fun=cost,
        x0=theta,
        method='L-BFGS-B',
        jac=grad,
        options={
            'maxiter': int(max_vqe_iterations - adam_iters),
            'ftol': 1e-10,
            'gtol': 1e-6,
            'maxls': 40,
            'disp': True
        }
    )

    theta_opt = result.x

    # -----------------------------
    # Step 6) Final state
    # -----------------------------
    final_state_vector = cudaq.get_state(vqe_ansatz_kernel, theta_opt, initial_state_np, n_layers, n_qubits)

    print(f"  [VQE Step] Complete.")
    print(f"  [VQE Step] Final Energy: {result.fun:.8f} Ha")

    return result.fun, final_state_vector