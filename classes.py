import utilities
import numpy as np
import networkx as nx


class ChannelNetwork:
    def __init__(self, graph, block_nodes, block_heights_from_surface, block_coeff_k, n_manning, y_ini_below_DEM, y_BC_below_DEM, Q_ini_value, Q_BC, channel_width) -> None:
        """This class has 2 purposes:
                1. get data from cwl network and store it as (properly ordered) nparrays
                2. Compute all things related to the canal topology and its impact to the discrete form of the PDE equations

        Args:
            graph ([type]): [description]
            block_nodes ([type]): [description]
            block_heights_from_surface ([type]): [description]
            block_coeff_k ([type]): [description]
            y_ini_below_DEM ([type]): [description]
            Q_ini_value ([type]): [description]
            q_value ([type]): [description]
        """
        # Unpack
        self.graph = graph
        self.n_manning = n_manning # manning coefficient
        self.y_BC_below_DEM = y_BC_below_DEM # downstream BC (m below DEM)
        self.Q_BC = Q_BC  # upstream BC in [units of dx]/ [units of dt]
        self.B = channel_width  # (rectangular) canal width in m

        # transpose is needed because of difference with NetworkX
        self.cnm = nx.adjacency_matrix(graph).T # canal network adjacency matrix
        self.n_nodes = self.cnm.shape[0]

        # In order to translate labels in the graph to positions in the arrays,
        # nodelist keeps track of position array <--> node label in graph
        # Also worth mentioning: graph.nodes is the default unpacking order used by networkx 
        self.nodelist = list(graph.nodes)
        # With this setup, the switch between array coords and graph labels works as follows:
        # 1) graph values -> array: use the method from_node_attribute_to_nparray()
        # 2) array -> node dictionary: use the method from_nparray_to_nodedict()

        # change labels in dem dictionary of nodes to positions in the array.
        self.dem = self.from_graph_attribute_to_nparray('DEM')
        self.q = self.from_graph_attribute_to_nparray('q')

        # Topology
        upstream_bool, downstream_bool = utilities.infer_extreme_nodes(
            self.cnm)  # BC nodes
        self.upstream_nodes = tuple(np.argwhere(upstream_bool).flatten())
        self.downstream_nodes = tuple(np.argwhere(downstream_bool).flatten())
        self.junctions = utilities.sparse_get_all_junctions(self.cnm)

        # Stuff for the system of equations in the Jacobian and F
        self.pos_eqs_to_remove = self.get_positions_of_eqs_to_remove()
        self.number_junction_eqs_to_add = self.count_total_junction_branches()
        self.check_equal_n_eqs_add_and_remove()
        # These go on top
        self.pos_of_junction_eqs_to_add = self.pos_eqs_to_remove[0:self.number_junction_eqs_to_add]
        self.pos_of_BC_eqs_to_add = self.pos_eqs_to_remove[self.number_junction_eqs_to_add:]

        # Node variables in nparray format
        self.B = channel_width * np.ones(self.n_nodes)
        self.y = self.dem - y_ini_below_DEM
        self.Q = Q_ini_value * np.ones(self.n_nodes)

        # Set y_BC and Q_BC inside the initial conditions
        self.y[np.array(self.downstream_nodes)] = self.dem[np.array(self.downstream_nodes)] - \
            self.y_BC_below_DEM
        self.Q[np.array(self.upstream_nodes)] = self.Q_BC

        # Block stuff
        # todo: In the future, block positions will be passed as attributes of the network.
        self.block_nodes = np.array(block_nodes)
        self.block_heights = np.array([self.dem[n] + block_heights_from_surface[i] for i,n in enumerate(block_nodes)])
        self.block_coeff_k = block_coeff_k

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
        number_BC_eqs_to_add = len(
            self.downstream_nodes) + len(self.upstream_nodes)
        total_number_eqs_to_add = self.number_junction_eqs_to_add + number_BC_eqs_to_add
        if total_number_eqs_to_add != len(self.pos_eqs_to_remove):
            raise ValueError(
                'Mismatch in number of eqs to add and to remove from J and F.')
        return 0

    def translate_from_label_in_graph_to_position_in_vector(self, list_node_labels):
        # Takes a list of ints specifying the labels of some nodes in the graph
        # and returns their positions in the vectorized of the eqs
        dict_from_label_in_graph_to_position_in_vector = {
            n_node: i for n_node, i in zip(self.nodelist, range(0, self.n_nodes))}
        return [dict_from_label_in_graph_to_position_in_vector[n_node] for n_node in list_node_labels]

    def from_nodedict_to_nparray(self, node_var):
        # Takes variable in nodedict and returns a properly ordered nparray
        return np.array([node_var[n] for n in self.nodelist])

    def from_nparray_to_nodedict(self, var_vector):
        # Takes numpy array that holds the value of a channel network variable and
        # returns it as a dictionary of nodes with original graph labels
        node_var = dict()
        for value, nodename in zip(var_vector, self.nodelist):
            node_var[nodename] = value
        return node_var

    def from_graph_attribute_to_nparray(self, attr_name):
        attr = nx.get_node_attributes(self.graph, attr_name)
        return np.array(list(attr.values()))


class GlobalParameters:
    def __init__(self, g, dt, dx, a, max_niter_newton, max_niter_inexact, ntimesteps, rel_tol, abs_tol, weight_A, weight_Q) -> None:
        self.g = g  # gravity
        self.dt = dt  # same units as g
        self.a = a  # weight parameter in Preissmann scheme
        self.dx = dx
        self.max_niter_newton = max_niter_newton
        self.max_niter_inexact = max_niter_inexact
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.weight_A = weight_A
        self.weight_Q = weight_Q
        self.ntimesteps = ntimesteps

        pass
