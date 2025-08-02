import torch
import cudaq
import numpy as np
from scipy.optimize import minimize
from tqdm import trange

# --- ADD THIS IMPORT ---
# Import the VMC sampler to be used for generating the initial state
from vmc_cal import efficient_parallel_sampler

@cudaq.kernel
def vqe_ansatz_kernel(thetas: list[float], initial_state: list[complex], n_electrons: int, n_qubits: int):
    # Initialize the qubit register directly from the passed-in numpy array.
    qubits = cudaq.qvector(initial_state)
    # for i in range(n_electrons):
    #     x(qubits[i])
    # Apply the UCCSD ansatz.
    cudaq.kernels.uccsd(qubits, thetas, n_electrons, n_qubits)

def run_vqe_fine_tuning(molecule_ham, nqs_model, n_qubits, n_electrons, vmc_params, max_vqe_iterations, device='cpu'):
    """
    Step 2: VQE Fine-tuning using a globally defined kernel.
    """
    print(f"  [VQE Step] Sampling {vmc_params['n_samples']} configurations from NQS using VMC sampler...")
    with torch.no_grad():
        samples = efficient_parallel_sampler(
            nqs_model,
            vmc_params['n_samples'] // vmc_params['n_chains'],
            vmc_params['n_chains'],
            n_qubits,
            vmc_params['burn_in_steps'],
            vmc_params['step_intervals'],
            device
        )
        unique_samples, _ = torch.unique(samples, dim=0, return_inverse=True)
        log_psi_unique = nqs_model.log_prob(unique_samples)
        amplitudes = torch.exp(log_psi_unique)

    full_state_vector = torch.zeros(2**n_qubits, dtype=torch.complex64, device=device)
    
    # Perform binary-to-decimal conversion using floating point tensors
    binary_float = (unique_samples + 1) / 2.0
    powers_of_two_float = (2**torch.arange(n_qubits - 1, -1, -1, device=device, dtype=torch.float32))
    int_indices = (binary_float @ powers_of_two_float).long()
    
    full_state_vector[int_indices] = amplitudes
    
    # Normalize the state vector and ensure it is a C-contiguous numpy array of the correct type.
    initial_state_np = np.ascontiguousarray(
        (full_state_vector / torch.norm(full_state_vector)).cpu().numpy(),
        dtype=np.complex64
    )

    print(initial_state_np)
    print(f"  [VQE Step] Constructed initial state from {len(amplitudes)} unique configurations.")

    # parameter_count = cudaq.kernels.uccsd_num_parameters(n_electrons, n_qubits)
    parameter_count = cudaq.kernels.num_hwe_parameters(n_electrons, n_qubits)
    
    # The cost function simply passes the numpy array and other integers to observe.
    def cost(theta):
        energy = cudaq.observe(vqe_ansatz_kernel, molecule_ham, theta, initial_state_np, n_electrons, n_qubits).expectation()
        print(f"[DEBUG] Energy at theta = {theta[:4]}...: {energy:.6f}")
        return energy

    x0 = np.random.normal(0, 2 * np.pi, parameter_count)
    
    print("  [VQE Step] Optimizing with COBYLA...")
    result = minimize(cost, x0, method='L-BFGS-B', options={'maxiter': max_vqe_iterations})

    # Get the final state vector from the optimized kernel.
    final_state_vector = cudaq.get_state(vqe_ansatz_kernel, result.x, initial_state_np, n_electrons, n_qubits)

    print(f"  [VQE Step] Complete.")
    print(f"  [VQE Step] Final State Vector: {final_state_vector}")
    print(f"  [VQE Step] Final Energy: {result.fun:.8f} Ha")

    return result.fun, final_state_vector

def generate_training_data_from_vqe(target_state_vector, n_qubits, n_samples, device='cpu'):
    """
    Step 3: Generate "Ground Truth" Data from VQE.
    """
    print("  [Data Gen Step] Generating supervised training data from VQE state...")
    probs_gpu = torch.from_numpy(np.abs(target_state_vector)**2).to(device)
    
    sampled_indices = torch.multinomial(probs_gpu, num_samples=n_samples, replacement=True)
    target_amplitudes = torch.from_numpy(target_state_vector).to(device)[sampled_indices]

    binary_repr = ((sampled_indices.unsqueeze(1) >> torch.arange(n_qubits - 1, -1, -1, device=device)) & 1).float()
    spin_configs = (binary_repr * 2 - 1).to(device)
    
    print(f"  [Data Gen Step] Generated {n_samples} (spin, amplitude) pairs.")
    return spin_configs, target_amplitudes

def run_nqs_supervised_training(nqs_model, spin_configs, target_amplitudes, n_epochs, device='cpu'):
    """
    Step 4: Supervised NQS Re-training.
    """
    print(f"  [NQS Retrain Step] Starting supervised training for {n_epochs} epochs...")
    optimizer = torch.optim.Adam(nqs_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    target_log_psi = torch.log(target_amplitudes)
    
    for epoch in trange(n_epochs, desc="  NQS Supervised"):
        optimizer.zero_grad()
        pred_log_psi = nqs_model.log_prob(spin_configs)
        loss = loss_fn(pred_log_psi.real, target_log_psi.real) + loss_fn(pred_log_psi.imag, target_log_psi.imag)
        loss.backward()
        optimizer.step()
    
    print("  [NQS Retrain Step] Complete.")