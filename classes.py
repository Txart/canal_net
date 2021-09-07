import utilities
import numpy as np

class ChannelNetworkParameters:
    def __init__(self, cnm, block_nodes, block_heights, block_coeff_k) -> None:
        # if type(cnm) != np.ndarray:
        #     raise TypeError(
        #         "The matrix must be a numpy array, this is required by some functions of this class.")

        self.cnm = cnm
        self.n_nodes = cnm.shape[0]
        upstream_bool, downstream_bool = utilities.infer_extreme_nodes(cnm)  # BC nodes
        self.upstream_nodes = tuple(np.argwhere(upstream_bool).flatten())
        self.downstream_nodes = tuple(np.argwhere(downstream_bool).flatten())
        self.junctions = utilities.sparse_get_all_junctions(cnm)

        self.block_nodes = block_nodes
        self.block_heights = block_heights
        self.block_coeff_k = block_coeff_k
        
        # Stuff for the system of equations in the Jacobian and F
        self.pos_eqs_to_remove = self.get_positions_of_eqs_to_remove()
        self.number_junction_eqs_to_add = self.count_total_junction_branches()
        self.check_equal_n_eqs_add_and_remove()
        self.pos_of_junction_eqs_to_add = self.pos_eqs_to_remove[0:self.number_junction_eqs_to_add] # These go on top
        self.pos_of_BC_eqs_to_add = self.pos_eqs_to_remove[self.number_junction_eqs_to_add:]
        
        pass
    
    def get_positions_of_eqs_to_remove(self):
        nodes_to_remove = []
        # Junction eqs
        for up_nodes, _ in self.junctions:
            for up_node in up_nodes:
                nodes_to_remove = nodes_to_remove + [2*up_node, 2*up_node+1]
        # BCs
        for down_node in self.downstream_nodes:
            nodes_to_remove = nodes_to_remove + [2*down_node, 2*down_node+1]
            
        return nodes_to_remove
        
    def count_total_junction_branches(self):
        count = 0
        for up_nodes, down_nodes in self.junctions:
            count = count + len(up_nodes) + len(down_nodes)
        return count   

    def check_equal_n_eqs_add_and_remove(self):
            number_BC_eqs_to_add = len(self.downstream_nodes) + len(self.upstream_nodes)
            total_number_eqs_to_add = self.number_junction_eqs_to_add + number_BC_eqs_to_add
            if total_number_eqs_to_add != len(self.pos_eqs_to_remove):
                raise ValueError('Mismatch in number of eqs to add and to remove from J and F.')
            return 0
        
        
    
class GlobalParameters:
    def __init__(self, g, dt, dx, a, n_manning, max_niter_newton, max_niter_inexact, ntimesteps, rel_tol, abs_tol, weight_A, weight_Q) -> None:
        self.g = g # gravity
        self.dt = dt # same units as g
        self.a = a # weight parameter in Preissmann scheme
        self.dx = dx
        self.n_manning = n_manning
        self.max_niter_newton = max_niter_newton
        self.max_niter_inexact = max_niter_inexact
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.weight_A = weight_A
        self.weight_Q = weight_Q
        self.ntimesteps = ntimesteps
        
        pass
