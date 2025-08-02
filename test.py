# import cudaq
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# import numpy as np

# # # Pass in a state from another kernel
# # c = [0.70710678 + 0j, 0., 0., 0.70710678]


# # @cudaq.kernel
# # def kernel_initial():
# #     q = cudaq.qvector(c)


# # state_to_pass = cudaq.get_state(kernel_initial)
# # print(state_to_pass)


# # @cudaq.kernel
# # def kernel(state: cudaq.State):
# #     q = cudaq.qvector(state)


# # state_to_pass = cudaq.get_state(kernel, state_to_pass)
# # print(state_to_pass)


# # # Single precision
# cudaq.set_target("nvidia")

# hydrogen_count = 2

# # Distance between the atoms in Angstroms.
# bond_distance = 1.26

# # Define a linear chain of Hydrogen atoms
# geometry = [('C', (0, 0, i * bond_distance)) for i in range(hydrogen_count)]

# molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
#     geometry, 'sto-3g', 1, 0)

# electron_count = data.n_electrons
# qubit_count = 2 * data.n_orbitals

# @cudaq.kernel
# def kernel(thetas: list[float]):

#     qubits = cudaq.qvector(qubit_count)

#     for i in range(electron_count):
#         x(qubits[i])

#     cudaq.kernels.uccsd(qubits, thetas, electron_count, qubit_count)


# parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count,
#                                                      qubit_count)

# # Define a function to minimize
# def cost(theta):

#     exp_val = cudaq.observe(kernel, molecule, theta).expectation()
#     print(f"Energy: {exp_val:.8f} Ha")

#     return exp_val


# exp_vals = []


# def callback(xk):
#     exp_vals.append(cost(xk))


# # Initial variational parameters.
# np.random.seed(42)
# x0 = np.random.normal(0, np.pi, parameter_count)

# # Use the scipy optimizer to minimize the function of interest
# result = minimize(cost,
#                   x0,
#                   method='COBYLA',
#                   callback=callback,
#                   options={'maxiter': 1})

from pyscf import gto, scf, fci

# Define the C2 molecule
mol = gto.Mole()
mol.atom = [['C', (0, 0, 0)], ['C', (0, 0, 1.26)]]
mol.basis = 'sto-3g'
mol.spin = 0  # singlet
mol.charge = 0
mol.build()

# Hartree-Fock calculation
mf = scf.RHF(mol)
hf_energy = mf.kernel()

# Full Configuration Interaction (FCI) for ground truth
fci_solver = fci.FCI(mf)
fci_energy, _ = fci_solver.kernel()

print("Hartree-Fock Energy:", hf_energy)
print("FCI Ground Truth Energy:", fci_energy)

