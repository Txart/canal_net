import numpy as np
import scipy.linalg

import math_preissmann
import utilities

def remove_multiple_up_branches(up_nodes, cnms):
    for up_node in up_nodes:
        cnms[:, up_node] = np.zeros(shape=cnms[:, 0].shape)
    return cnms


def remove_multiple_down_branches(down_nodes, cnms):
    for down_node in down_nodes:
        cnms[down_node] = np.zeros(shape=cnms[0].shape)
    return cnms


def simplify_graph_by_removing_junctions(channel_network_params):
    # Separate into branches with no junctions
    cnm_simple = channel_network_params.cnm.copy()
    for up_nodes, down_nodes in channel_network_params.junctions:
        if len(up_nodes) > 1:
            cnm_simple = remove_multiple_up_branches(up_nodes, cnm_simple)
        if len(down_nodes) > 1:
            cnm_simple = remove_multiple_down_branches(down_nodes, cnm_simple)

    return cnm_simple


def SS_computation(channel_network, general_params):
    y = channel_network.y
    Q = channel_network.Q
    B = channel_network.B
    q = channel_network.q
    for i in range(general_params.max_niter_newton):
        print(i)
        jacoSS = math_preissmann.build_SS_jacobian(y, Q, B, general_params, channel_network)
        FuSS = math_preissmann.build_SS_F(y, Q, q, B, general_params, channel_network)

        x = scipy.linalg.solve(jacoSS, -FuSS)

        x_Q, x_y = utilities.deinterweave_array(x)
        
        if np.linalg.norm(x) < general_params.rel_tol*np.linalg.norm(utilities.interweave_vectors(y, Q)) + general_params.abs_tol:
            print('\n>>> Inexact Newton-Rhapson converged after ', i, ' iterations')
            break

        y = y + general_params.weight_A*x_y
        Q = Q + general_params.weight_Q*x_Q

        if np.any(np.isnan(y) | np.isnan(Q)):
            print(
                '\n>>> y: ', y, ' x_y: ', x_y, ' \n >>> Q: ',Q, ' x_Q: ', x_Q,)
            raise ValueError('Nan at some point of Inexact Newton-Raphson')

    returny,Q