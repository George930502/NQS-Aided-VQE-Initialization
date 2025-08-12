import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cudaq
from models import FFNN
from moleculars import get_pyscf_results, MOLECULE_DATA
from vmc_cal import efficient_parallel_sampler, stochastic_reconfiguration_update, local_energy_batch
from vqe_module import run_vqe_fine_tuning, generate_training_data_from_vqe, run_nqs_supervised_training
from tqdm import trange
from debug import *

def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def plot_results(results, seps, config):
    method_name = "FFNN+VQE"
    molecule_name = config['molecule_name']
    basis = MOLECULE_DATA[molecule_name]['basis']
    plt.figure(figsize=(12, 8))
    results_VMC = results.pop(method_name)
    for method, values in results.items():
        plt.plot(seps, values, linestyle='--', label=method)
    energies = [res[0] for res in results_VMC]
    errors = [res[1] for res in results_VMC]
    plt.errorbar(seps, energies, yerr=errors, marker='o', linestyle='-', label=method_name, capsize=5)
    plt.xlabel("Separation (Å)"), plt.ylabel("Energy (Hartree)")
    plt.title(f"{molecule_name} Potential Energy Dissociation Curve ({basis})")
    plt.legend(), plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(), plt.show()

if __name__ == '__main__':
    config = load_config()
    hybrid_params = config['hybrid_params']
    molecule_choice = config['molecule_name']
    vmc_params = config['vmc_params']
    ffnn_params = config['ffnn_params']
    hew_params = config['hew_params']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudaq.set_target("nvidia")
    print(f"Using PyTorch device: {device}, CUDA-Q target: {cudaq.get_target().name}")

    geom = MOLECULE_DATA[molecule_choice]['geometry']
    curve_config = config['dissociation_curve']
    seps = np.linspace(curve_config['start_separation'], curve_config['end_separation'], curve_config['points'])
    base_dist = np.linalg.norm(np.array(geom[0][1]) - np.array(geom[1][1]))
    scales = seps / base_dist if base_dist > 0 else np.ones_like(seps)

    results = {'HF': [], 'FCI': [], 'CCSD': [], 'CCSD(T)': [], "FFNN+VQE": []}
    
    for i, scale in enumerate(scales):
        label = f"{seps[i]:.3f} Å"
        print(f"\n--- Calculating for {molecule_choice} @ {label} ---")

        mol_pyscf, hf_e, fci_e, ccsd_e, ccsd_t_e, qham_of = get_pyscf_results(molecule_choice, scale)
        mol_geom_for_cudaq = [(str(atom), tuple(pos)) for atom, pos in mol_pyscf.atom]
        molecule_ham, data = cudaq.chemistry.create_molecular_hamiltonian(mol_geom_for_cudaq, MOLECULE_DATA[molecule_choice]['basis'])
        results['HF'].append(hf_e); results['FCI'].append(fci_e); results['CCSD'].append(ccsd_e); results['CCSD(T)'].append(ccsd_t_e)

        n_orbitals = mol_pyscf.nao_nr() * 2
        n_hidden = int(n_orbitals * ffnn_params['alpha'])
        nqs_model = FFNN(n_orbitals, n_hidden, ffnn_params['n_layers'], device=device)

        # --- HYBRID ALGORITHM WITH FEEDBACK LOOP ---
        
        # Step 1: Initial NQS Pre-training
        print(f"[Step 1] Starting NQS pre-training for {hybrid_params['nqs_pretrain_epochs']} epochs...")
        for ep in trange(hybrid_params['nqs_pretrain_epochs'], desc="NQS Pre-training"):
            samples = efficient_parallel_sampler(
                nqs_model, vmc_params['n_samples'] // vmc_params['n_chains'], vmc_params['n_chains'], n_orbitals,
                vmc_params['burn_in_steps'], vmc_params['step_intervals'], device=device
            )
            stochastic_reconfiguration_update(
                nqs_model, samples, qham_of, lr=vmc_params['learning_rate'], reg=vmc_params['sr_regularization'], device=device
            )
            if (ep + 1) % 10 == 0:
                eval_local_energies = local_energy_batch(nqs_model, samples, qham_of, device)
                eval_mean = eval_local_energies.mean().item()
                eval_std = eval_local_energies.std().item() / np.sqrt(len(eval_local_energies))
                print(f"[Epoch {ep + 1}] Eval Energy: {eval_mean:.6f} ± {eval_std:.6f} Ha")

        print("[Step 1] NQS pre-training finished.")

        run_alignment_test(nqs_model, qham_of, n_orbitals)

        final_energy = 0.0
        # Start the iterative feedback loop
        for loop in range(hybrid_params['feedback_loops']):
            print(f"\n--- Starting Feedback Loop {loop + 1}/{hybrid_params['feedback_loops']} ---")
            
            # Step 2: VQE Fine-tuning
            vqe_energy, target_state_vector = run_vqe_fine_tuning(
                molecule_ham, nqs_model, n_orbitals, data.n_electrons, hew_params['n_layers'],
                vmc_params, qham_of, hybrid_params['vqe_max_iterations'], device
            )
            final_energy = vqe_energy # Store the energy from this loop

            # Step 3: Generate Training Data
            spin_configs, target_amplitudes = generate_training_data_from_vqe(
                target_state_vector, n_orbitals, hybrid_params['vqe_num_samples'], device
            )

            # Step 4: Supervised NQS Re-training
            run_nqs_supervised_training(
                nqs_model, spin_configs, target_amplitudes, hybrid_params['nqs_retrain_epochs'], device
            )

        print("\n--- All feedback loops complete ---")
        results["FFNN+VQE"].append((final_energy, 0.0))

        final_e_val, final_e_std_val = results["FFNN+VQE"][-1]
        print("\n--- Results Summary ---")
        print(f"  Hartree-Fock: {hf_e:.6f} Ha")
        print(f"  FCI:          {fci_e:.6f} Ha")
        print(f"  CCSD:         {ccsd_e:.6f} Ha")
        print(f"  CCSD(T):      {ccsd_t_e:.6f} Ha")
        print(f"  FFNN+VQE:     {final_e_val:.8f} ± {final_e_std_val:.8f} Ha")

    plot_results(results, seps, config)