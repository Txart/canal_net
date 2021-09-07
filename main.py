# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:34:31 2021

@author: 03125327
"""

# %% Imports
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from tqdm import tqdm  # progress bar
import time

import utilities

# %%


def laplacians(directed_adj_matrix):
    """
    Returns the 'modified Laplacians', i.e., the matrix operators for the
    Lax-Friederichs scheme
    """
    D_out = np.diag(np.sum(directed_adj_matrix, axis=0))
    D_in = np.diag(np.sum(directed_adj_matrix, axis=1))
    A_out = directed_adj_matrix.T
    # and A_in = directed_adj_matrix

    L_in_minus = directed_adj_matrix - D_in
    L_in_plus = np.abs(L_in_minus)
    L_out_minus = A_out - D_out
    L_out_plus = np.abs(L_out_minus)

    return L_in_minus, L_in_plus, L_out_minus, L_out_plus


def compute_laplacian_from_adjacency(adj_matrix):
    if np.any(adj_matrix != adj_matrix.T):
        raise ValueError(
            'the matrix must be symmetric, i.e., must be the adj matrix of an undirected graph')
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix

    return laplacian


def signed_laplacian(directed_adj_matrix):
    A_signed = directed_adj_matrix - directed_adj_matrix.T
    degree_signed = np.diag(
        np.sum(directed_adj_matrix, axis=1) - np.sum(directed_adj_matrix, axis=0))

    return A_signed + degree_signed


def sparse_laplacian_matrices(L_adv):
    sL_adv = scipy.sparse.csr_matrix(L_adv)
    positive_sL_adv = np.abs(sL_adv)
    sL_adv_t = sL_adv.T
    positive_sL_adv_t = np.abs(sL_adv_t)

    return sL_adv, positive_sL_adv, sL_adv_t, positive_sL_adv_t

# %% Numerics


def friction_slope(V, Y, B):
    return n_manning**2 * V**2/(wetted_radius(Y, B)**(4/3))


def wetted_radius(Y, B):
    return Y*B/(2*Y + B)


# %% PARAMETERS and DATA


# Topology. Canal network
n_nodes = 10
graph_name = 'Y'
cnm = utilities.choose_graph(graph_name, n_nodes=n_nodes)
upstream_bool, downstream_bool = utilities.infer_extreme_nodes(cnm)  # BC nodes
# BLOCK_LOCATION = downstream_bool.copy()
BLOCK_LOCATION = np.zeros(n_nodes, dtype=bool)

# Physics
B_scale = 7
B = B_scale * np.ones(n_nodes)  # (rectangular) canal width in m
g = 9.8  # m/s
# g = 9.8 * (60*60*24)**2 # m/day
n_manning = 0

# Numerics
ndays = 1000
dx = 50
# dt = 1 *60 # Same time units as g
dt = 1e-3  # same time units as g
niter = int(ndays/dt)
# For newton method
max_niter_newton = 100000


# GIS data
bottom = utilities.choose_bottom(graph_name, n_nodes)


# %% Vectorial Lax-Friederichs
explicit_laxFriederichs = False


def lax_friederichs(dx, dt, L, L_signed, u, F, G):
    return 0.5*(-L @ u + dt/dx * L_signed @ F) + dt*G


def F_of_V(V, A):
    return 0.5*V**2 + g*(A/B + bottom)


def G_of_V(V, A):
    return - g*friction_slope(V, A/B, B)


if explicit_laxFriederichs:
    # Compute advection Laplacian
    # L_in_minus, L_in_plus, L_out_minus, L_out_plus = laplacians(cnm)
    L = compute_laplacian_from_adjacency(cnm + cnm.T)
    L_signed = signed_laplacian(cnm)

    # Initial and boundary conditions
    # BC
    upstream_bool, downstream_bool = utilities.infer_extreme_nodes(
        cnm)  # Get BC nodes
    block_height = 2.2 * BLOCK_LOCATION

    # IC
    # Y_ini = 1 + max(bottom) - bottom  # m above canal bottom
    Y_ini = 2 + bottom

    q = 0.0 * np.ones(n_nodes)  # Lateral inflow of water
    # q[upstream_bool] = 1
    V_ini = np.zeros(n_nodes)
    V_upstream = 0
    V_ini[upstream_bool] = V_upstream
    # TODO: solve steady state for initial condition of V
    # for i in range(n_nodes-1):
    #     V_ini[i+1] = ((q[i]+q[i+1])/2*dx + V_ini[i]*A_ini[i])/A_ini[i+1]

    # sparsify
    # L  = scipy.sparse.csr_matrix(L)
    # L_signed   = scipy.sparse.csr_matrix(L_signed)

    V = V_ini.copy()
    A = B * Y_ini.copy()

    # Simulate. Store solution in terms of Q = AV and Y = A/B
    V_sol = [0]*niter
    A_sol = [0]*niter
    for t in tqdm(range(niter)):

        A_sol[t] = A
        V_sol[t] = V

        F_V = F_of_V(V, A)
        G_V = G_of_V(V, A)
        # F_V = 0.001 * np.ones(shape=V.shape)
        # G_V = 0.001 * np.ones(shape=V.shape)

        A = A + lax_friederichs(dx, dt, L, L_signed, A, V*A, q)
        V = V + lax_friederichs(dx, dt, L, L_signed, V, F_V, G_V)

        # (Time varying) BC
        # V[upstream_bool] = V_upstream
        # V[downstream_bool] = 0.
        # A[upstream_bool] = A_ini[0]

        # Downstream BC

        # If there is a dam, set limit to height in the dam
        # k=1
        # V[BLOCK_LOCATION] = 0
        # for i,block in enumerate(BLOCK_LOCATION):
        #     if block == True:
        #         if Y[i] > block_height[i]:
        #             k = 4.
        #             V[i] = (k * np.sqrt(g) * (Y[i] - block_height[i])**1.5)/(Y[i]*B[i]) # Eq. 8.101 from Szymkiewicz's book

    Y_sol = A_sol/B
    Q_sol = [V_sol[t] * A_sol[t] for t in range(niter)]

# %% Implicit backwards Euler


def create_jacobian(dt, dx, A, V, g, n_manning, B, A_signed, degree_in_minus_out, dtype='double'):

    VA = utilities.interweave_arrays(dt/(2*dx) * V, dt/(2*dx) * A, dtype=dtype)
    VA2 = utilities.interweave_arrays(
        dt/(2*dx) * g/B, dt/(2*dx) * V, dtype=dtype)

    # Build sign matrix from incidence matrix
    temp = utilities.interweave_arrays(A_signed, A_signed, dtype=dtype)
    # interweave by columns by transposing
    signs = utilities.interweave_arrays(temp.T, temp.T, dtype=dtype).T

    # diagonal block
    temp = utilities.interweave_arrays(
        degree_in_minus_out, degree_in_minus_out, dtype=dtype)

    # jVjA and jVjV extra terms
    bidiag = utilities.tridiag(np.ones(2*n_nodes-1, dtype=dtype), np.ones(2 *
                                                                          n_nodes, dtype=dtype), np.zeros(2*n_nodes-1, dtype=dtype))
    bidiag[0::2] = np.zeros(2*n_nodes, dtype=dtype)
    jVjA_term = -4/3*dt*g*n_manning**2*V**2*(2/B + B/A)**(1/3)*B/(A**2)
    # matrix with signs and weights in the diagonal
    signed_diag = utilities.interweave_arrays(temp.T, temp.T, dtype=dtype)
    jVjV_term = 2*dt*g*n_manning**2*(2/B + B/A)**(4/3)*V
    extra_jV_terms = utilities.interweave_arrays(
        jVjA_term, jVjV_term, dtype=dtype)
    extra_jV_terms_matrix = np.zeros(shape=(2*n_nodes, 2*n_nodes), dtype=dtype)
    extra_jV_terms_matrix[1::2] = extra_jV_terms
    extra_jV_terms_matrix = np.multiply(extra_jV_terms_matrix, bidiag)

    # Finally, construct Jacobian
    jacobian = np.zeros(shape=(2*n_nodes, 2*n_nodes), dtype=dtype)
    # VA terms
    jacobian[0::2] = VA
    jacobian[1::2] = VA2
    VA_terms_weights = signs + signed_diag
    jacobian = np.multiply(jacobian, VA_terms_weights)

    # extra terms in every even row
    jacobian = jacobian + extra_jV_terms_matrix

    # -1 in the diagonal
    # -1 in the diagonal
    jacobian = jacobian - np.diag(np.ones(2*n_nodes, dtype=dtype))

    return jacobian


def F_of_A(A, A_previous, V, q, dt, dx, L_prime):
    return A_previous - A + dt/(2*dx) * L_prime @ (A*V) + dt*q


def F_of_V(V, V_previous, A, dt, dx, g, B, bottom, L_prime):
    return V_previous - V + dt/(2*dx)*L_prime @ (0.5*V**2 + g/B*A + g*bottom) + dt*g*friction_slope(V, A/B, B)


def adaptive_weight(x, norm_u):
    """
    Input is the change in solution x as solved from Newton system
    norm_u is the norm of the unknown we are solving
    """
    w_ceiling = 1e-2
    w_floor = 1e-3

    weight = w_ceiling * (1 - x/norm_u)
    return max(weight, w_floor)


if explicit_laxFriederichs == False:
    tstart = time.time()
    # Network topology matrices for jacobian
    A_signed = cnm - cnm.T
    # incoming - outgoing degree
    degree_in_minus_out = np.diag(np.sum(A_signed, axis=1))
    L_prime = A_signed + degree_in_minus_out

    # Initial and boundary conditions
    # BC
    upstream_bool, downstream_bool = utilities.infer_extreme_nodes(
        cnm)  # Get BC nodes
    block_height = 2.2 * BLOCK_LOCATION

    # IC
    # Y_ini = 1 + max(bottom) - bottom  # m above canal. Horizontal water
    Y_ini = 2 + bottom  # sloped water

    q = 0.0 * np.ones(n_nodes)  # Lateral inflow of water
    # q[upstream_bool] = 1
    V_ini = 1.0 * np.ones(n_nodes)
    V_upstream = 1.0
    V_downstream = 0.2
    V_ini[upstream_bool] = V_upstream
    V_ini[downstream_bool] = V_downstream
    # TODO: solve steady state for initial condition of V
    # for i in range(n_nodes-1):
    #     V_ini[i+1] = ((q[i]+q[i+1])/2*dx + V_ini[i]*A_ini[i])/A_ini[i+1]

    # sparsify
    # L  = scipy.sparse.csr_matrix(L)
    # L_signed   = scipy.sparse.csr_matrix(L_signed)

    V = V_ini.copy()
    V_previous = V_ini.copy()
    A = B*Y_ini.copy()
    A_previous = B*Y_ini.copy()

    # Change to single precision (32 bit) numpy arrays
    # A = A.astype(dtype='single')
    # A_previous = A_previous.astype(dtype='single')
    # V = V.astype(dtype='single')
    # V_previous = V_previous.astype(dtype='single')
    # B = B.astype(dtype='single')
    # A_signed = A_signed.astype(dtype='single')
    # degree_in_minus_out = degree_in_minus_out.astype(dtype='single')

    # Newton-Rhapson
    rel_tol = 1e-7
    abs_tol = 1e-7

    downstream_BC_mode = 'constant_velocity'  # It can also be 'constant_height'

    def newtonRaphson(A, A_previous, V, V_previous):

        for i in range(max_niter_newton):
            jacobian = create_jacobian(
                dt, dx, A, V, g, n_manning, B, A_signed, degree_in_minus_out)

            F_A = F_of_A(A, A_previous, V, q, dt, dx, L_prime)
            F_V = F_of_V(V, V_previous, A, dt, dx, g, B, bottom, L_prime)

            # BCs
            # Upstream
            #  A: Constant height BC
            for up in np.where(upstream_bool)[0]:
                jacobian[up] = np.zeros(shape=2*A.shape[0])
                jacobian[up, up] = 1.0
                F_A[up] = 0.
                # V: from characteristics

                def celerity(A, B):
                    return np.sqrt(g*A/B)
                c = celerity(A, B)
                V_S = (V_previous[up] - dt/dx*(c[up+1]*V_previous[up] - c[up]*V_previous[up+1])) / (
                    1 - dt/dx*(V_previous[up] - V_previous[up+1] - c[up] + c[up+1]))
                c_S = (c[up] + V_S*dt/dx*(c[up] - c[up+1])) / \
                    (1 + dt/dx*(c[up] - c[up+1]))
                y_S = A_previous[up]/B[up] + dt/dx * \
                    (V_S-c_S)*(A_previous[up]/B[up] - A_previous[up+1]/B[up+1])
                jacobian[up+1] = np.zeros(shape=2*V.shape[0])
                jacobian[up+1, up] = g/(c_S * B[up])
                jacobian[up+1, up+1] = -1
                sf_S = 0.5 * (friction_slope(V_previous[up+1], A_previous[up+1]/B[up+1], B[up+1]) + friction_slope(
                    V_previous[up], A_previous[up]/B[up], B[up]))
                F_V[up] = -V[up] + V_S - g/c_S*y_S + dt*g * \
                    ((A_previous[up]/B[up] - A_previous[up+1]/B[up+1]) /
                     dx - sf_S) + g/c_S * A[up]/B[up]

            # Downstream:
            V_R = (V_previous[-1] - dt/dx*(c[-2]*V_previous[-1] - c[-1]*V_previous[-2])) / (
                1 + dt/dx*(V_previous[-1] - V_previous[-2] + c[-1] - c[-2]))
            c_R = c[-1] - V_R*dt/dx*(c[-1] - c[-2])/(1 + dt/dx*(c[-1] - c[-2]))
            y_R = A_previous[-1]/B[-1] - dt/dx * \
                (V_R + c_R)*(A_previous[-1]/B[-1] - A_previous[-2]/B[-2])
            sf_R = 0.5*(friction_slope(V_previous[-2], A_previous[-2]/B[-2], B[-2]) + friction_slope(
                V_previous[-1], A_previous[-1]/B[-1], B[-1]))

            if downstream_BC_mode == 'constant_height':
                jacobian[-2] = np.zeros(shape=2*A.shape[0])
                jacobian[-2, -2] = 1.0
                F_A[-1] = 0.
                # V: from characteristics

                jacobian[-1] = np.zeros(shape=2*V.shape[0])
                jacobian[-1, -1] = -1.
                jacobian[-1, -2] = -g/(c_R*B[-1])

                F_V[-1] = -V[-1] + V_R + g/c_R*y_R + g*dt * \
                    ((A_previous[-2]/B[-2] - A_previous[-1]/B[-1]) /
                     dx - sf_R) - g/c_R*A[-1]/B[-1]

            elif downstream_BC_mode == 'constant_velocity':
                jacobian[-1] = np.zeros(shape=2*V.shape[0])
                jacobian[-1, -1] = 1.
                F_V[-1] = 0
                # A: from characteristics
                jacobian[-2] = np.zeros(shape=2*A.shape[0])
                jacobian[-2, -2] = -1.0
                jacobian[-2, -1] = -B[-1]*c_R/g
                F_A[-1] = -A[-1] + B[-1]*(c_R/g*(V_R + g/c_R*y_R + g*dt*(A_previous[-2]/B[-2] - A_previous[-1]/B[-1]) /
                                                 dx - V[-1]))

            else:
                raise ValueError('Downstream BC incorrectly specified')

            # Put together RHS
            F_u = utilities.interweave_arrays(F_A, F_V)

            # x = scipy.sparse.linalg.solve(jacobian, -F_u)
            x = np.linalg.solve(jacobian, -F_u)
            x_A, x_V = utilities.deinterweave_array(x)

            # weight_A = adaptive_weight(np.linalg.norm(x_A), 0.1*np.linalg.norm(A))
            # weight_V = adaptive_weight(np.linalg.norm(x_V), 0.1*np.linalg.norm(V))
            weight_A = 1e-3
            weight_V = 1e-3
            # print(
            #     f"norm(x_A), norm(A): {np.linalg.norm(x_A), 0.5*np.linalg.norm(A)}")
            # print(f"weight_A, weight_V: {weight_A, weight_V}")

            A = A + weight_A*x_A
            V = V + weight_V*x_V

            if np.linalg.norm(x) < rel_tol*np.linalg.norm(utilities.interweave_arrays(A_previous, V_previous)) + abs_tol:
                print(f'\n>>> Newton-Rhapson converged after {i} iterations')
                return A, V
            elif np.any(np.isnan(A)):
                print('\n>>> NaN at some point')
                raise ValueError('Nan at some point of Newton-Raphson')

        return A, V

    # try:
    #     A_sol, V_sol = newtonRaphson(A, A_previous, V, V_previous)
    # except:
    #     print('An exception occurred in the Newton method')
    A_sol = [0]*ndays
    V_sol = [0]*ndays
    for timestep in tqdm(range(ndays)):
        A, V = newtonRaphson(A, A_previous, V, V_previous)
        A_previous = A.copy()
        V_previous = V.copy()
        A_sol[timestep] = A
        V_sol[timestep] = V

    print(f'\n It took {time.time() - tstart} seconds')

    Y_sol = [A_sol[t]/B for t in range(ndays)]
    Q_sol = [V_sol[t] * A_sol[t] for t in range(ndays)]

# %% Plot, animations and prints
plotOpt = True
if plotOpt:
    import plotting

    # Plot initial water height
    if explicit_laxFriederichs == True:
        niter_to_plot = min(1000, niter)
        total_iterations = niter
    else:
        niter_to_plot = ndays
        total_iterations = ndays

    alfa = min(1/niter_to_plot, 0.1)
    xx = np.arange(0, dx*n_nodes, dx)

    plotting.plot_water_height(xx, Y_ini, bottom)
    plotting.plot_velocity(xx, V_ini)
    plotting.plot_all_Qs(xx, Q_sol, alfa, niter_to_plot, total_iterations)
    plotting.plot_Qs_animation(
        xx, Q_sol, total_iterations, niter_to_plot, 'output/Q_full_StVenants.mp4')
    plotting.plot_all_Ys(xx, Y_sol, bottom, alfa,
                         total_iterations, niter_to_plot)

    plotting.plot_Ys_animation(
        xx, Y_sol, bottom, block_height, BLOCK_LOCATION, alfa, total_iterations, niter_to_plot, 'output/Y_full_StVenants.mp4')
    plotting.plot_conservations(Y_sol, V_sol, total_iterations)
