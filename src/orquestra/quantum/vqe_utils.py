from openfermion.utils import uccsd_singlet_paramsize

def build_circuit_template(ansatz_type, n_mo, n_alpha, n_beta, transformation='Jordan-Wigner',
                           fermion_generator=None, spin_ordering='interleaved', 
                           n_qubits=None, ordering=None, layers=None,
                           has_rx_layer=False, elementary=False):
    """Constructs a circuit template for an ansatz.

    Args:
        n_mo (int): number of molecular orbitals
        n_alpha (int): number of alpha electrons
        n_beta (int): number of beta electrons
        ansatz_type (str): which ansatz to use. Currently only 'singlet UCCSD' is supported.
        transformation (str): which Fermion transformation to use
        fermion_generator (openfermion.FermionOperator): Fermion generator for a general UCC 
            ansatz
        threshold (float): a threshold used in the selection of the operators composing the ansatz
        spin_ordering (str): how spins are ordered in the qubit register, allow us
                             to create correct HF state. 'interleaved': spins alpha and
                             beta are alternating ; 'blocks': all spin alpha, followed  
                             by all spin beta
        n_qubits (int): The number of qubits. If None, it is taken to be 2 * n_mo
        ordering (list): Ordering of the lattice sites, each pair of numbers indicates
                                  which qubits the spin-up and spin-down lattice site corespond to.
                                  Even indices are spin-up qubits and odd are spin-down qubits
        layers (zquantum.core.circuit.CircuitLayers): 
                                Object describing the layout of layers of 2-qubit gate blocks.
                                The layers attribute is a list of list of tuples where each tuple
                                contains the indices of the qubits where the gate block should be 
                                aplied. The layers are applied in the order, looping over the list,
                                until there are no more parameters.
        has_rx_layer (bool): whether to include the layer of Rx rotations at the beginning of the 
            ansatz.
        elementary (bool): if True, decomposes the U1ex and U2ex gates into elementary gates. Useful
                           if they are not defined on the device/simulator

    Returns:
        dict: dictionary describing the ansatz.
    """

    if n_qubits is None and n_mo is None:
        raise Exception("n_qubits or n_mo must be provided for all Ansatze")
    if n_qubits is None:
        n_qubits = 2 * n_mo

    if ansatz_type == "singlet UCCSD":
        if(n_alpha != n_beta):
            raise RuntimeError('Number of alpha and beta electrons must be equal for a singlet UCCSD ansatz')
        if spin_ordering != 'interleaved':
            raise RuntimeError('Only the interleaved spin ordering is implemented for singlet UCCSD')
        n_params = [ uccsd_singlet_paramsize(n_qubits=n_qubits,
                                           n_electrons=n_alpha+n_beta) ]
        ansatz =    {'ansatz_type': 'singlet UCCSD',
                    'ansatz_module': 'zquantum.vqe.ansatzes.ucc',
                    'ansatz_func' : 'build_singlet_uccsd_circuit',
                    'ansatz_kwargs' : {
                            'n_mo' : n_mo,
                        'n_electrons' : n_alpha+n_beta,
                        'transformation' : transformation},
                     'n_params': n_params}
        return(ansatz)

    else:
        raise RuntimeError('Ansatz "{}" not implemented'.format(ansatz_type))
