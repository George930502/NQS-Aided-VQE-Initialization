import torch
import cudaq
import numpy as np
from scipy.optimize import minimize
from tqdm import trange
from cudaq import SpinOperator

# --- ADD THIS IMPORT ---
# Import the VMC sampler to be used for generating the initial state
from vmc_cal import efficient_parallel_sampler, local_energy_batch

@cudaq.kernel
def vqe_ansatz_kernel_test(thetas: list[float], initial_state: list[complex], n_electrons: int, n_qubits: int):
    # Initialize the qubit register directly from the passed-in numpy array.
    qubits = cudaq.qvector(initial_state)
    # for i in range(n_electrons):
    #     x(qubits[i])
    # Apply the UCCSD ansatz.
    # cudaq.kernels.uccsd(qubits, thetas, n_electrons, n_qubits)

@cudaq.kernel
def prepare_only(initial_state: list[complex]):
    _ = cudaq.qvector(initial_state)  

@cudaq.kernel
def vqe_ansatz_kernel_test1(thetas: list[float], initial_state: list[complex], n_layers: int, n_qubits: int):
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
            rx(thetas[theta_idx], qubits[q])
            theta_idx += 1

        # CNOT entangling layer (even-odd pairwise)
        for q in range(0, n_qubits - 1, 1):
            x.ctrl(qubits[q], qubits[q + 1])
        # for q in range(1, n_qubits - 1, 2):
        #     x.ctrl(qubits[q], qubits[q + 1])
        # for q in range(n_qubits // 2):
        #     x.ctrl(qubits[q], qubits[n_qubits - 1 - q])
    
    for q in range(n_qubits):
        ry(thetas[theta_idx], qubits[q])
        theta_idx += 1
        rx(thetas[theta_idx], qubits[q])
        theta_idx += 1

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
            ry(thetas[theta_idx], qubits[q])
            theta_idx += 1

        # CNOT entangling layer (even-odd pairwise)
        # for q in range(0, n_qubits - 1, 2):
        #     x.ctrl(qubits[q], qubits[q + 1])
        # for q in range(1, n_qubits - 1, 2):
        #     x.ctrl(qubits[q], qubits[q + 1])
        for q in range(n_qubits // 2):
            z.ctrl(qubits[q], qubits[n_qubits - 1 - q])

    # Optional: final RY-RZ layer after all entangling blocks
    for q in range(n_qubits):
        rz(thetas[theta_idx], qubits[q])
        theta_idx += 1
        ry(thetas[theta_idx], qubits[q])
        theta_idx += 1
        rz(thetas[theta_idx], qubits[q])
        theta_idx += 1

# -------- 直接用 NQS 枚舉 2^n 建完整複振幅（little_endian=True）--------
def build_full_state_from_model(nqs_model, n_qubits):
    dev = next(nqs_model.parameters()).device
    with torch.no_grad():
        N = 1 << n_qubits
        idxs = torch.arange(N, device=dev, dtype=torch.long)
        # 小端序：col j 對應 qubit j（最低位）
        bits = ((idxs.unsqueeze(1) >> torch.arange(0, n_qubits, device=dev)) & 1).float()
        # σ = 1 - 2*bit  （|0>→+1, |1>→-1）
        sigmas = 1.0 - 2.0 * bits
        # 你的 FFNN.log_prob 回傳的是「log ψ」（複數）
        log_psi = nqs_model.log_prob(sigmas)           # complex
        amps = torch.exp(log_psi).to(torch.complex64)  # ψ(s)
        psi = amps / torch.linalg.norm(amps)
        return np.ascontiguousarray(psi.detach().cpu().numpy().astype(np.complex64))

def spsa_optimization(cost_fn, x0, max_iters=100, a=0.2, c=0.1, avg_grad=True, avg_samples=3):
    """
    Basic (non-adaptive) SPSA: constant step size a and perturbation size c.
    """
    theta = x0.copy()
    n_params = len(theta)

    for _ in trange(max_iters, desc="  SPSA"):
        ak = a       # <- 固定，不隨迭代改變
        ck = c       # <- 固定，不隨迭代改變

        grad = np.zeros_like(theta)

        # 可選的梯度平均，減少方差
        reps = (avg_samples if avg_grad else 1)
        for _ in range(reps):
            delta = 2 * np.random.randint(0, 2, size=n_params) - 1  # ±1
            theta_plus  = theta + ck * delta
            theta_minus = theta - ck * delta
            y_plus  = cost_fn(theta_plus)
            y_minus = cost_fn(theta_minus)
            grad += (y_plus - y_minus) / (2.0 * ck * delta)

        grad /= reps
        theta = theta - ak * grad

    return theta


def adaptive_spsa_optimization(cost_fn, x0, max_iters=100, a=0.2, c=0.1, alpha=0.602, gamma=0.101, avg_grad=True, avg_samples=3):
    """
    Adaptive SPSA with optional gradient averaging.
    """
    theta = x0.copy()
    n_params = len(theta)

    for k in trange(max_iters, desc="  Adaptive SPSA"):
        ak = a / ((k + 1) ** alpha)
        ck = c / ((k + 1) ** gamma)

        grad = np.zeros_like(theta)

        for _ in range(avg_samples if avg_grad else 1):
            delta = 2 * np.random.randint(0, 2, size=n_params) - 1
            theta_plus = theta + ck * delta
            theta_minus = theta - ck * delta
            y_plus = cost_fn(theta_plus)
            y_minus = cost_fn(theta_minus)
            grad += (y_plus - y_minus) / (2.0 * ck * delta)

        grad /= avg_samples
        theta = theta - ak * grad

    return theta

@cudaq.kernel
def vqe_ansatz_kernel_safe(thetas: list[float], initial_state: list[complex],
                           n_layers: int, n_qubits: int):
    q = cudaq.qvector(initial_state)
    if n_layers <= 0:
        return
    tidx = 0
    for _ in range(n_layers):
        # 單量子位旋轉（θ=0 ⇒ I）
        for i in range(n_qubits):
            ry(thetas[tidx], q[i]); tidx += 1
            rz(thetas[tidx], q[i]); tidx += 1
            ry(thetas[tidx], q[i]); tidx += 1
        # 參數化 ZZ（θ=0 ⇒ I）
        for i in range(n_qubits - 1):
            x.ctrl(q[i], q[i+1])
            rz(thetas[tidx], q[i+1]); tidx += 1
            x.ctrl(q[i], q[i+1])
    # 最後一層單量子位旋轉（θ=0 ⇒ I）
    for i in range(n_qubits):
        rz(thetas[tidx], q[i]); tidx += 1
        ry(thetas[tidx], q[i]); tidx += 1
        rz(thetas[tidx], q[i]); tidx += 1

# -------- 修正版 VQE：初始 cost = NQS 能量 --------
def run_vqe_fine_tuning(molecule_ham, nqs_model, n_qubits, n_electrons,  # n_electrons 可不使用
                        n_layers, vmc_params, qham_of, max_vqe_iterations, device='cpu'):
    # 1) 兩端共用同一份哈密頓量來源
    molecule_ham = SpinOperator(qham_of)

    # 2) 用「完整 NQS 狀態」當初始態（little_endian=True）
    initial_state_np = build_full_state_from_model(nqs_model, n_qubits)
    print(initial_state_np)

    # 3) 檢查：不施加 gate 的 <H>（這應等於你剛剛量到的 -0.28869066 Ha）
    E0 = cudaq.observe(prepare_only, molecule_ham, initial_state_np).expectation()
    print(f"[Init] Energy of NQS state (CUDA-Q) = {E0:.8f} Ha")

    # （可選）如果仍想看抽樣估計：
    samples = efficient_parallel_sampler(
            nqs_model,
            vmc_params['n_samples'] // vmc_params['n_chains'],
            vmc_params['n_chains'],
            n_qubits,
            vmc_params['burn_in_steps'],
            vmc_params['step_intervals'],
            device)

    print("E_vmc (MC mean):", local_energy_batch(nqs_model, samples, qham_of, device).mean().item())

    # 4) 參數數量（與 kernel 定義一致）
    parameter_count = n_layers * (3 * n_qubits) + 3 * n_qubits
    print(f"[VQE] parameter_count = {parameter_count}")

    # 5) cost 與 θ=0（應該回報 E0）
    def cost(theta):
        energy = cudaq.observe(
            vqe_ansatz_kernel_safe, molecule_ham,
            theta, initial_state_np, n_layers, n_qubits
        ).expectation()
        print(f"[DEBUG] Energy: {energy:.6f}")
        return energy

    theta0 = np.zeros(parameter_count, dtype=float)
    E_at_theta0 = cost(theta0)
    print(f"[Check] cost(theta0) = {E_at_theta0:.8f} Ha")  # 應 ≈ E0

    # x0 = np.random.normal(0, 2 * np.pi, parameter_count)

    print("  [VQE Step] Optimizing with COBYLA...")
    result = minimize(cost, theta0, method='COBYLA', options={'maxiter': max_vqe_iterations})

    # SPSA optimization
    # print("  [VQE Step] Optimizing with SPSA...")
    # theta_opt = adaptive_spsa_optimization(cost, x0, max_iters=max_vqe_iterations)
    # Get the final state vector from the optimized kernel.
    # final_state_vector = cudaq.get_state(vqe_ansatz_kernel, result.x, initial_state_np, n_electrons, n_qubits)
    final_state_vector = cudaq.get_state(vqe_ansatz_kernel, result.x, initial_state_np, n_layers, n_qubits)
    # final_state_vector = cudaq.get_state(vqe_ansatz_kernel, theta_opt, initial_state_np, n_layers, n_qubits)

    print(f"  [VQE Step] Complete.")
    # print(f"  [VQE Step] Final State Vector: {final_state_vector}")
    print(f"  [VQE Step] Final Energy: {result.fun:.8f} Ha")
    # print(f"  [VQE Step] Final Energy: {cost(theta_opt):.8f} Ha")
 
    return result.fun, final_state_vector
    # return cost(theta_opt), final_state_vector

    # 6) 優化（你原本的 SPSA）
    # theta_opt = adaptive_spsa_optimization(cost, theta0, max_iters=max_vqe_iterations)

    # # 7) 輸出最後能量與態
    # E_final = cost(theta_opt)
    # final_state_vector = cudaq.get_state(
    #     vqe_ansatz_kernel_test1, theta_opt, initial_state_np, n_layers, n_qubits
    # )
    # print(f"[VQE] final energy   = {E_final:.8f} Ha")
    # return E_final, final_state_vector

# def run_vqe_fine_tuning(molecule_ham, nqs_model, n_qubits, n_electrons, n_layers, vmc_params, qham_of, max_vqe_iterations, device='cpu'):
#     """
#     Step 2: VQE Fine-tuning using a globally defined kernel.
#     """
#     print(f"  [VQE Step] Sampling {vmc_params['n_samples']} configurations from NQS using VMC sampler...")
#     with torch.no_grad():
#         samples = efficient_parallel_sampler(
#             nqs_model,
#             vmc_params['n_samples'] // vmc_params['n_chains'],
#             vmc_params['n_chains'],
#             n_qubits,
#             vmc_params['burn_in_steps'],
#             vmc_params['step_intervals'],
#             device
#         )
        
#         unique_samples, _ = torch.unique(samples, dim=0, return_inverse=True)
#         # print(unique_samples)
#         log_psi_unique = nqs_model.log_prob(unique_samples)
#         # print(log_psi_unique)
#         amplitudes = torch.exp(log_psi_unique)
#         # print(amplitudes)

#     full_state_vector = torch.zeros(2**n_qubits, dtype=torch.complex64, device=device)
    
#     # Perform binary-to-decimal conversion using floating point tensors
#     # binary_float = (unique_samples + 1) / 2.0
#     # powers_of_two_float = (2**torch.arange(n_qubits - 1, -1, -1, device=device, dtype=torch.float32))
#     # int_indices = (binary_float @ powers_of_two_float).long()

#     bits = (1.0 - unique_samples.to(torch.float32)) / 2.0      # [B, n_qubits] in {0,1}
#     powers_of_two = 2 ** torch.arange(0, n_qubits, device=bits.device, dtype=torch.float32)
#     int_indices = (bits.flip(1) @ powers_of_two).long()        # [B]
#     print(bits)
#     print(int_indices)

#     full_state_vector[int_indices] = amplitudes
    
#     # Normalize the state vector and ensure it is a C-contiguous numpy array of the correct type.
#     initial_state_np = np.ascontiguousarray(
#         (full_state_vector / torch.norm(full_state_vector)).cpu().numpy(),
#         dtype=np.complex64
#     )

#     molecule_ham = SpinOperator(qham_of)
#     print("qham: ", qham_of)
#     print("molecule_ham: ", molecule_ham)

#     eval_local_energies = local_energy_batch(nqs_model, samples, qham_of, device)
#     eval_mean = eval_local_energies.mean().item()
#     print(f"Eval Energy: {eval_mean:.6f} Ha")
    
#     E_nqs_in_cudaq = cudaq.observe(prepare_only, molecule_ham, initial_state_np).expectation()
#     print(f"[Check] Energy from CUDA-Q on NQS state = {E_nqs_in_cudaq:.8f} Ha")

#     print(f"  [VQE Step] Constructed initial state from {len(amplitudes)} unique configurations.")

#     # parameter_count = cudaq.kernels.uccsd_num_parameters(n_electrons, n_qubits)
#     parameter_count = 3 * n_qubits * (n_layers + 1)
#     print(f"  [VQE Step] Number of parameters in ansatz: {parameter_count}")
    
#     # The cost function simply passes the numpy array and other integers to observe.
#     def cost(theta):
#         # energy = cudaq.observe(vqe_ansatz_kernel, molecule_ham, theta, initial_state_np, n_electrons, n_qubits).expectation()
#         energy = cudaq.observe(vqe_ansatz_kernel, molecule_ham, theta, initial_state_np, n_layers, n_qubits).expectation()        
#         print(f"[DEBUG] Energy: {energy:.6f}")
#         return energy

#     x0 = np.random.normal(0, 2 * np.pi, parameter_count)
#     # x0 = np.random.normal(0, 2 * np.pi)
    
#     # COBYLA optimization
#     # print("  [VQE Step] Optimizing with COBYLA...")
#     # result = minimize(cost, x0, method='COBYLA', options={'maxiter': max_vqe_iterations})

#     # SPSA optimization
#     print("  [VQE Step] Optimizing with SPSA...")
#     theta_opt = adaptive_spsa_optimization(cost, x0, max_iters=max_vqe_iterations)
#     # Get the final state vector from the optimized kernel.
#     # final_state_vector = cudaq.get_state(vqe_ansatz_kernel, result.x, initial_state_np, n_electrons, n_qubits)
#     # final_state_vector = cudaq.get_state(vqe_ansatz_kernel, result.x, initial_state_np, n_layers, n_qubits)
#     final_state_vector = cudaq.get_state(vqe_ansatz_kernel, theta_opt, initial_state_np, n_layers, n_qubits)

#     print(f"  [VQE Step] Complete.")
#     # print(f"  [VQE Step] Final State Vector: {final_state_vector}")
#     # print(f"  [VQE Step] Final Energy: {result.fun:.8f} Ha")
#     print(f"  [VQE Step] Final Energy: {cost(theta_opt):.8f} Ha")

#     # return result.fun, final_state_vector
#     return cost(theta_opt), final_state_vector


def generate_training_data_from_vqe(target_state_vector, n_qubits, n_samples, device='cpu'):
    """
    Step 3: Generate "Ground Truth" Data from VQE.
    """
    print("  [Data Gen Step] Generating supervised training data from VQE state...")
    target_state_vector_np = np.array(target_state_vector)
    probs_gpu = torch.from_numpy(np.abs(target_state_vector_np)**2).to(device)
    
    sampled_indices = torch.multinomial(probs_gpu, num_samples=n_samples, replacement=True)
    target_amplitudes = torch.from_numpy(target_state_vector_np).to(device)[sampled_indices]

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
    
    for _ in trange(n_epochs, desc="  NQS Supervised"):
        optimizer.zero_grad()
        pred_log_psi = nqs_model.log_prob(spin_configs)
        loss = loss_fn(pred_log_psi.real, target_log_psi.real) + loss_fn(pred_log_psi.imag, target_log_psi.imag)
        loss.backward()
        optimizer.step()
    
    print("  [NQS Retrain Step] Complete.")