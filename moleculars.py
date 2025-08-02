from pyscf import gto, scf, fci, cc
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from openfermion import MolecularData

# --- Data from the provided table ---
MOLECULE_DATA = {
    'H2': {
        'basis': 'STO-3G',
        'geometry': [['H', (0, 0, 0)], ['H', (0, 0, 0.734)]],
    },
    'LiH': {
        'basis': 'STO-3G',
        'geometry': [['Li', (0, 0, 0)], ['H', (0, 0, 1.548)]],
    },
    'C2': {
        'basis': 'STO-3G',
        'geometry': [['C', (0, 0, 0)], ['C', (0, 0, 1.26)]],
    },
    'N2': {
        'basis': 'STO-3G',
        'geometry': [['N', (0, 0, 0)], ['N', (0, 0, 1.19)]],
    },
    'NH3': {
        'basis': 'STO-3G',
        'geometry': [
            ['N', (0, 0, 0.149)],
            ['H', (0, 0.947, -0.348)],
            ['H', (0.821, -0.474, -0.348)],
            ['H', (-0.821, -0.474, -0.348)]
        ],
    },
    'H2O_STO-3G': {
        'basis': 'STO-3G',
        'geometry': [
            ['O', (0, 0, 0.137)],
            ['H', (0, 0.769, -0.546)],
            ['H', (0, -0.769, -0.546)]
        ],
    },
    'H2O_6-31G': {
        'basis': '6-31G',
        'geometry': [
            ['O', (0, 0, 0.113)],
            ['H', (0, 0.795, -0.454)],
            ['H', (0, -0.795, -0.454)]
        ],
    }
}

def get_pyscf_results(molecule_name, dist_scale=1.0):
    """
    Builds the molecule, runs HF/FCI/CCSD/CCSD(T), and constructs the JW-mapped qubit Hamiltonian.
    Returns: mol, hf_e, fci_e, ccsd_e, ccsd_t_e, qubit_hamiltonian
    """
    data = MOLECULE_DATA[molecule_name]
    geom = data['geometry']

    # scale diatomic separations
    if len(geom) == 2:
        atom1, pos1 = geom[0]
        atom2, pos2 = geom[1]
        scaled_pos2 = tuple([p * dist_scale for p in pos2])
        scaled_geom = [[atom1, pos1], [atom2, scaled_pos2]]
    else:
        scaled_geom = geom

    # PySCF molecule
    mol = gto.Mole()
    mol.atom = scaled_geom
    mol.basis = data['basis']
    mol.build(verbose=0)

    # Hartree-Fock
    mf = scf.RHF(mol).run(verbose=0)
    hf_e = mf.e_tot

    # FCI
    try:
        fci_solver = fci.FCI(mf)
        fci_e, _ = fci_solver.kernel()
    except Exception:
        fci_e = hf_e

    # CCSD / CCSD(T)
    try:
        ccsd_solver = cc.CCSD(mf)
        ccsd_solver.kernel()
        ccsd_e = ccsd_solver.e_tot
        ccsd_t_e = ccsd_e + ccsd_solver.ccsd_t()
    except Exception:
        ccsd_e = hf_e
        ccsd_t_e = hf_e

    # Build OpenFermion MolecularData and run PySCF integrals
    mol_of = MolecularData(
        geometry=scaled_geom,
        basis=data['basis'],
        multiplicity=1,
        charge=0
    )
    mol_of = run_pyscf(mol_of, run_scf=True, run_fci=False, verbose=False)
    
    # Get the molecular Hamiltonian (fermionic) and map to qubits
    fermion_ham = mol_of.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(fermion_ham)
    qubit_ham = jordan_wigner(fermion_op)

    return mol, hf_e, fci_e, ccsd_e, ccsd_t_e, qubit_ham