import torch
import numpy as np
import warnings
from scipy.sparse.linalg import LinearOperator, cg

def efficient_parallel_sampler(model, n_samples_per_chain, n_chains, n_spins, burn_in, step_interval, device='cpu'):
    current_configs = torch.randint(0, 2, (n_chains, n_spins), device=device, dtype=torch.float32) * 2 - 1
    with torch.no_grad():
        log_probs_current = model.log_prob(current_configs)

        for _ in range(burn_in):
            flip_indices = torch.randint(0, n_spins, (n_chains,), device=device)
            props = current_configs.clone()
            props[torch.arange(n_chains), flip_indices] *= -1
            log_probs_prop = model.log_prob(props)
            acceptance_prob = torch.exp(2 * (log_probs_prop.real - log_probs_current.real))
            accept_mask = torch.rand(n_chains, device=device) < acceptance_prob
            current_configs[accept_mask] = props[accept_mask]
            log_probs_current[accept_mask] = log_probs_prop[accept_mask]

        samples = torch.zeros(n_samples_per_chain * n_chains, n_spins, device=device, dtype=torch.float32)

        for i in range(n_samples_per_chain):
            for _ in range(step_interval):
                flip_indices = torch.randint(0, n_spins, (n_chains,), device=device)
                props = current_configs.clone()
                props[torch.arange(n_chains), flip_indices] *= -1
                log_probs_prop = model.log_prob(props)
                acceptance_prob = torch.exp(2 * (log_probs_prop.real - log_probs_current.real))
                accept_mask = torch.rand(n_chains, device=device) < acceptance_prob
                current_configs[accept_mask] = props[accept_mask]
                log_probs_current[accept_mask] = log_probs_prop[accept_mask]
            samples[i * n_chains: (i+1) * n_chains] = current_configs

    return samples

def local_energy_batch(model, samples, qubit_hamiltonian, device='cpu'):
    with torch.no_grad():
        unique_samples, inverse_indices = torch.unique(samples, dim=0, return_inverse=True)
        log_psi_unique = model.log_prob(unique_samples)
        E_loc_unique = torch.zeros(unique_samples.shape[0], dtype=torch.complex64, device=device)
        for term, coeff in qubit_hamiltonian.terms.items():
            if not term:
                E_loc_unique += coeff
                continue
            samples_prime = unique_samples.clone()
            phase = torch.ones(unique_samples.shape[0], dtype=torch.complex64, device=device)
            for idx, pauli in term:
                vals = unique_samples[:, idx]
                if pauli == 'Z': 
                    phase *= vals
                else:
                    samples_prime[:, idx] *= -1
                    if pauli == 'Y': 
                        phase *= 1j * vals
            log_psi_prime_unique = model.log_prob(samples_prime)
            E_loc_unique += coeff * phase * torch.exp(log_psi_prime_unique - log_psi_unique)
        return E_loc_unique[inverse_indices].real

def stochastic_reconfiguration_update(model, samples, qubit_ham, lr, reg, device='cpu'):
    params = list(model.parameters())
    n_p = sum(p.numel() for p in params)
    O_list = []
    log_psi_batch = model.log_prob(samples)

    for i in range(samples.shape[0]):
        model.zero_grad()
        log_psi_batch[i].real.backward(retain_graph=True)
        grad_real = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
        model.zero_grad()
        log_psi_batch[i].imag.backward(retain_graph=True)
        grad_imag = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
        O_list.append(grad_real + 1j * grad_imag)

    O = torch.stack(O_list)
    local_es = local_energy_batch(model, samples, qubit_ham, device=device).to(torch.complex64)
    F = torch.einsum('ki,k->i', O.conj(), local_es) / samples.shape[0]
    F -= O.conj().mean(dim=0) * local_es.mean()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)

            def S_matvec(v_np):
                v = torch.from_numpy(v_np.astype(np.complex64)).to(device)
                Ov = torch.einsum('ki,i->k', O, v)
                term1 = torch.einsum('ki,k->i', O.conj(), Ov) / samples.shape[0]
                O_mean_dot_v = torch.dot(O.mean(dim=0), v)
                term2 = O.conj().mean(dim=0) * O_mean_dot_v
                return (term1 - term2 + reg * v).cpu().numpy().astype(np.complex64)
            
            S_operator = LinearOperator(shape=(n_p, n_p), matvec=S_matvec, dtype=np.complex64)
            delta_np, info = cg(S_operator, F.cpu().numpy().astype(np.complex64))
            delta = F if info != 0 else torch.from_numpy(delta_np.astype(np.complex64)).to(device)
            
    except (RuntimeWarning, RuntimeError):
        delta = F

    with torch.no_grad():
        idx = 0
        for p in params:
            p -= lr * delta[idx: idx+p.numel()].reshape(p.shape).real
            idx += p.numel()

# import torch
# import numpy as np
# import warnings
# from scipy.sparse.linalg import LinearOperator, cg

# def _efficient_update(theta_current, sigma_current, flip_idx, model):
#     """Calculates the change in log_prob efficiently after a single spin flip."""
#     sigma_flipped = sigma_current[torch.arange(sigma_current.shape[0]), flip_idx]
#     update_term = 2 * model.W[flip_idx] * sigma_flipped.unsqueeze(1)
#     theta_prop = theta_current - update_term
    
#     log_cosh_prop = torch.sum(torch.log(torch.cosh(theta_prop)), dim=1)
#     log_cosh_current = torch.sum(torch.log(torch.cosh(theta_current)), dim=1)
#     d_log_psi = -2 * model.a[flip_idx] * sigma_flipped + (log_cosh_prop - log_cosh_current)
    
#     acceptance_prob = torch.exp(2 * d_log_psi.real)
#     return acceptance_prob, theta_prop

# def efficient_parallel_sampler(model, n_samples_per_chain, n_chains, n_spins, burn_in, step_interval, device='cpu'):
#     """Generates samples using parallel MCMC samplers with efficient updates."""
#     current_configs = current_configs = (torch.randint(0, 2, (n_chains, n_spins), device=device) * 2 - 1).to(torch.cfloat)
    
#     with torch.no_grad():
#         theta_current = torch.einsum('bi,ij->bj', current_configs, model.W) + model.b

#         for _ in range(burn_in):
#             flip_indices = torch.randint(0, n_spins, (n_chains,), device=device)
#             acceptance_prob, theta_prop = _efficient_update(theta_current, current_configs, flip_indices, model)
#             accept_mask = torch.rand(n_chains, device=device) < acceptance_prob
#             current_configs[accept_mask, flip_indices[accept_mask]] *= -1
#             theta_current[accept_mask] = theta_prop[accept_mask]

#         samples = torch.zeros(n_samples_per_chain * n_chains, n_spins, device=device)
#         sample_count = 0
#         total_steps = n_samples_per_chain * step_interval
        
#         for i in range(total_steps):
#             flip_indices = torch.randint(0, n_spins, (n_chains,), device=device)
#             acceptance_prob, theta_prop = _efficient_update(theta_current, current_configs, flip_indices, model)
#             accept_mask = torch.rand(n_chains, device=device) < acceptance_prob
#             current_configs[accept_mask, flip_indices[accept_mask]] *= -1
#             theta_current[accept_mask] = theta_prop[accept_mask]

#             if (i + 1) % step_interval == 0:
#                 start_idx = sample_count * n_chains
#                 end_idx = start_idx + n_chains
#                 samples[start_idx:end_idx] = current_configs.real
#                 sample_count += 1
                
#     return samples

# def non_parallel_sampler(model, n_samples, n_spins, burn_in, step_interval, device='cpu'):
#     """Generates samples using a single MCMC chain with efficient updates."""
#     samples = torch.zeros(n_samples, n_spins, device=device)
#     current_config = torch.randint(0, 2, (1, n_spins), device=device, dtype=torch.float32) * 2 - 1
    
#     with torch.no_grad():
#         theta_current = torch.einsum('bi,ij->bj', current_config, model.W) + model.b

#         total_burn_in_steps = burn_in * n_spins
#         for _ in range(total_burn_in_steps):
#             flip_idx = torch.randint(0, n_spins, (1,), device=device)
#             acceptance_prob, theta_prop = _efficient_update(theta_current, current_config, flip_idx, model)
#             if torch.rand(1, device=device) < acceptance_prob:
#                 current_config[0, flip_idx] *= -1
#                 theta_current = theta_prop

#         sample_count = 0
#         total_steps = n_samples * step_interval
#         for i in range(total_steps):
#             flip_idx = torch.randint(0, n_spins, (1,), device=device)
#             acceptance_prob, theta_prop = _efficient_update(theta_current, current_config, flip_idx, model)
#             if torch.rand(1, device=device) < acceptance_prob:
#                 current_config[0, flip_idx] *= -1
#                 theta_current = theta_prop
            
#             if (i + 1) % step_interval == 0:
#                 samples[sample_count] = current_config.real
#                 sample_count += 1
    
#     return samples

# def local_energy_batch(model, samples, qubit_hamiltonian, device='cpu'):
#     """
#     Compute the local energy for a batch of samples, optimized by caching unique configurations.
#     """
#     with torch.no_grad():
#         # Find unique samples, their inverse mapping, and counts. This is the core of the optimization.
#         unique_samples, inverse_indices = torch.unique(samples, dim=0, return_inverse=True)
#         n_unique = unique_samples.shape[0]
        
#         # Calculate log_psi only for unique samples
#         log_psi_unique = model.log_prob(unique_samples)
        
#         E_loc_unique = torch.zeros(n_unique, dtype=torch.complex64, device=device)

#         for term, coeff in qubit_hamiltonian.terms.items():
#             if not term:  # Identity term
#                 E_loc_unique += coeff
#                 continue

#             samples_prime = unique_samples.clone()
#             phase = torch.ones(n_unique, dtype=torch.complex64, device=device)
            
#             for idx, pauli in term:
#                 vals = unique_samples[:, idx]
#                 if pauli == 'Z':
#                     phase *= vals
#                 else:  # X or Y
#                     samples_prime[:, idx] *= -1
#                     if pauli == 'Y':
#                         phase *= 1j * vals
            
#             log_psi_prime_unique = model.log_prob(samples_prime)
#             amp_ratio = torch.exp(log_psi_prime_unique - log_psi_unique)
#             E_loc_unique += coeff * phase * amp_ratio
        
#         # Expand the unique local energies back to the full sample size using the inverse map
#         E_loc = E_loc_unique[inverse_indices]
        
#     return E_loc.real

# def stochastic_reconfiguration_update(model, samples, qubit_ham, lr, reg, device='cpu'):
#     """
#     Updates model parameters using the Conjugate Gradient (CG) iterative solver for the SR linear system.
#     """
#     n_samples = samples.shape[0]
#     params = list(model.parameters())
#     n_p = sum(p.numel() for p in params)

#     samples_complex = samples.to(dtype=torch.cfloat)
    
#     # Compute analytical log-derivative vectors O_k
#     theta = torch.einsum('bi,ij->bj', samples_complex, model.W) + model.b
#     tanh_theta = torch.tanh(theta)
    
#     O_a = samples_complex
#     O_b = tanh_theta
#     O_W = torch.einsum('bi,bj->bij', samples_complex, tanh_theta).view(n_samples, -1)
    
#     O = torch.cat([O_a, O_b, O_W], dim=1)
    
#     # Define the matrix-vector product for the S matrix without forming S
#     def S_matvec(v_np):
#         """Computes the matrix-vector product S@v for a given vector v."""
#         v = torch.from_numpy(v_np).to(device)
#         Ov = torch.einsum('ki,i->k', O, v)
#         term1 = torch.einsum('ki,k->i', O.conj(), Ov) / n_samples
#         O_mean_dot_v = torch.dot(O.mean(dim=0), v)
#         term2 = O.conj().mean(dim=0) * O_mean_dot_v
#         reg_v = reg * v
#         result = term1 - term2 + reg_v
#         return result.detach().cpu().numpy()

#     # Create a SciPy LinearOperator
#     S_operator = LinearOperator(shape=(n_p, n_p), matvec=S_matvec, dtype=np.complex64)

#     # Compute force vector F
#     local_es = local_energy_batch(model, samples, qubit_ham, device=device).to(torch.cfloat)
#     F = torch.einsum('ki,k->i', O.conj(), local_es) / n_samples
#     F -= O.conj().mean(dim=0) * local_es.mean()
#     F_np = F.detach().cpu().numpy()
    
#     # Solve S * delta = F using Conjugate Gradient (CG)
#     try:
#         # Catch RuntimeWarnings that happen inside the solver
#         with warnings.catch_warnings():
#             warnings.filterwarnings('error', category=RuntimeWarning)
            
#             def S_matvec(v_np):
#                 v = torch.from_numpy(v_np.astype(np.complex64)).to(device)
#                 Ov = torch.einsum('ki,i->k', O, v)
#                 term1 = torch.einsum('ki,k->i', O.conj(), Ov) / n_samples
#                 O_mean_dot_v = torch.dot(O.mean(dim=0), v)
#                 term2 = O.conj().mean(dim=0) * O_mean_dot_v
#                 reg_v = reg * v
#                 result = term1 - term2 + reg_v
#                 return result.detach().cpu().numpy().astype(np.complex64)

#             S_operator = LinearOperator(shape=(n_p, n_p), matvec=S_matvec, dtype=np.complex64)
#             F_np = F.detach().cpu().numpy().astype(np.complex64)
            
#             delta_np, info = cg(S_operator, F_np)
            
#             if info != 0:
#                 print(f"CG solver did not converge (info={info}). Falling back to Gradient Descent.")
#                 delta = F # Use the force vector as the update direction
#             else:
#                 delta = torch.from_numpy(delta_np.astype(np.complex64)).to(device)

#     except (RuntimeWarning, RuntimeError) as e:
#         print(f"SR solver failed with numerical error: {e}. Falling back to Gradient Descent.")
#         delta = F # Fallback to using the force vector (gradient)

#     # Apply the update
#     with torch.no_grad():
#         idx = 0
#         for p in params:
#             num = p.numel()
#             update = delta[idx:idx+num].reshape(p.shape).real
#             p -= lr * update
#             idx += num