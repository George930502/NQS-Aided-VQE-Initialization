import numpy as np
import torch
import cudaq
from cudaq import spin, SpinOperator

# --------- 小工具：取得模型裝置 ----------
def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')

# --------- 只載入狀態，不加任何門 ----------
@cudaq.kernel
def prepare_only(initial_state: list[complex]):
    _ = cudaq.qvector(initial_state)

# --------- 用 NQS 建完整 2^n 狀態向量（自動跟隨模型裝置） ----------
def build_full_state_from_model(nqs_model, n_qubits, little_endian=True):
    dev = model_device(nqs_model)
    with torch.no_grad():
        N = 1 << n_qubits
        idxs = torch.arange(N, device=dev, dtype=torch.long)

        if little_endian:
            bits = ((idxs.unsqueeze(1) >> torch.arange(0, n_qubits, device=dev)) & 1).float()
        else:
            bits = ((idxs.unsqueeze(1) >> torch.arange(n_qubits - 1, -1, -1, device=dev)) & 1).float()

        # σ ∈ {+1, -1}，Z 本徵值
        sigmas = 1.0 - 2.0 * bits  # [2^n, n_qubits], float32 on dev

        # 你的 FFNN.log_prob 回傳的是「log ψ」（複數）
        log_psi = nqs_model.log_prob(sigmas)                 # complex on dev
        amps = torch.exp(log_psi).to(torch.complex64)        # ψ(s)
        psi = amps / torch.linalg.norm(amps)

        # 丟給 CUDA-Q 必須在 CPU / numpy
        return np.ascontiguousarray(psi.detach().cpu().numpy().astype(np.complex64))

# --------- 檢查 <Z_j>：CUDA-Q vs 直接由 |ψ|^2 ----------
def check_z_expectations(nqs_model, n_qubits, little_endian=True):
    dev = model_device(nqs_model)
    psi_np = build_full_state_from_model(nqs_model, n_qubits, little_endian)

    # CUDA-Q 端 <Z_j>
    z_cudaq = [cudaq.observe(prepare_only, spin.z(j), psi_np).expectation()
               for j in range(n_qubits)]

    # 直接由 |ψ|^2 計 <Z_j>
    with torch.no_grad():
        N = 1 << n_qubits
        probs = torch.from_numpy((np.abs(psi_np) ** 2)).to(dev)
        idxs = torch.arange(N, device=dev, dtype=torch.long)
        if little_endian:
            bits = ((idxs.unsqueeze(1) >> torch.arange(0, n_qubits, device=dev)) & 1).float()
        else:
            bits = ((idxs.unsqueeze(1) >> torch.arange(n_qubits - 1, -1, -1, device=dev)) & 1).float()
        sigmas = 1.0 - 2.0 * bits
        z_nqs = (probs.unsqueeze(1) * sigmas).sum(dim=0).detach().cpu().numpy().tolist()

    print(f"\n[Z check] little_endian={little_endian}")
    for j in range(n_qubits):
        print(f"  <Z{j}>  CUDA-Q = {z_cudaq[j]: .8f} ,   from |ψ|^2 = {z_nqs[j]: .8f}")

# --------- 用完整 ψ 計 <H>（不靠抽樣） ----------
def energy_cudaq_on_full_psi(molecule_ham, nqs_model, n_qubits, little_endian=True):
    psi_np = build_full_state_from_model(nqs_model, n_qubits, little_endian)
    return cudaq.observe(prepare_only, molecule_ham, psi_np).expectation()

# --------- 主測試：傳入 nqs_model、qham_of、n_qubits ----------
def run_alignment_test(nqs_model, qham_of, n_qubits):
    # 兩端共用同一份 OpenFermion QubitOperator
    molecule_ham = SpinOperator(qham_of)

    # 列出模型裝置，避免搞混
    print("[Device] nqs_model on:", model_device(nqs_model))

    # 兩種端序都試
    for little in (True, False):
        check_z_expectations(nqs_model, n_qubits, little_endian=little)
        E = energy_cudaq_on_full_psi(molecule_ham, nqs_model, n_qubits, little_endian=little)
        print(f"[H check] little_endian={little}  ->  <H>_CUDA-Q = {E:.8f} Ha")

    print("\n挑選讓每個 <Z_j> 與 <H> 都最一致的端序，後續 VQE 就用那個設定。")

# ======== 使用方式 ========
# 假設你已有：
#   nqs_model : 你的 FFNN（log ψ 複數），請確認它已放在你想用的裝置上 (CPU/GPU)
#   qham_of   : OpenFermion 的 QubitOperator（你之前已生成）
#   n_qubits  : 量子位數（H2 常見是 4）
#
# 直接呼叫：
#   run_alignment_test(nqs_model, qham_of, n_qubits)
#
# 若你在 GPU 上訓練：
#   nqs_model = nqs_model.to('cuda')
#   再執行 run_alignment_test(...)，本程式會自動跟隨模型裝置建張量。
