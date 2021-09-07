# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:34:31 2021

@author: 03125327
"""

#%% Imports
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time
from tqdm import tqdm

from numba import jit

#%%

def infer_extreme_nodes(directed_adj_matrix):
    """
    Infer what nodes are the beginning and end of canals from adjacency matrix.
    Last nodes of canals  are identified by having no outgoing edges
    First nodes of canals have no incoming edges

    Parameters
    ----------
    adj_matrix : numpy array
        Adjacency matrix of the canal network graph

    Returns
    -------
    end_nodes_bool : boolean numpy array
        True where nodes are last nodes of canals
    first_nodes_bool : boolean numpy array
        True where nodes are first nodes of canals

    """
    # Infer neumann and Diri nodes from adj matrix
    end_nodes_bool = np.sum(directed_adj_matrix, axis=0) == 0 # Boundary values below are conditional on this boolean mask
    first_nodes_bool = np.sum(directed_adj_matrix, axis=1) == 0 
    # in case the summing over the sparse matrix changes the numpy array shape
    end_nodes_bool = np.ravel(end_nodes_bool) 
    first_nodes_bool = np.ravel(first_nodes_bool)
   
    return first_nodes_bool, end_nodes_bool

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
        raise ValueError('the matrix must be symmetric, i.e., must be the adj matrix of an undirected graph')
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix
    
    return laplacian

def signed_laplacian(directed_adj_matrix):
        A_signed = directed_adj_matrix - directed_adj_matrix.T
        degree_signed = np.diag(np.sum(directed_adj_matrix, axis=1) - np.sum(directed_adj_matrix, axis=0))
        
        return A_signed + degree_signed

def sparse_laplacian_matrices(L_adv):
    sL_adv = scipy.sparse.csr_matrix(L_adv)
    positive_sL_adv = np.abs(sL_adv)
    sL_adv_t = sL_adv.T
    positive_sL_adv_t = np.abs(sL_adv_t)
    
    return sL_adv, positive_sL_adv, sL_adv_t, positive_sL_adv_t

#%% Numerics
    
def lax_friederichs(dx, dt, L, L_signed, u, F, G):
    return 0.5*(-L @ u + dt/dx* L_signed @ F) + dt*G

@jit(nopython=True)
def friction_slope(V, Y):
    return n_manning**2 * V**2/(wetted_radius(Y)**(4/3))

@jit(nopython=True)
def wetted_radius(Y):
    return Y*B/(2*Y + B)

@jit(nopython=True)
def F_of_V(V, Y):
    return 0.5*V**2 + g*(Y + bottom)

@jit(nopython=True)
def G_of_V(V, Y):
    return - g*friction_slope(V,Y)



#%%
def choose_graph(graph_name, n_nodes=5):
    """
    A small library of pre-defined networks.
    """
        
    if graph_name == 'line':
        graph = np.diag(np.ones(n_nodes-1), k=-1)
        
    elif graph_name == 'Y':
        nn = int(n_nodes/3) # nodes per branch

        graph= np.block([
                        [np.diag(np.ones(nn-1), k=-1), np.zeros((nn,nn)), np.zeros((nn,nn+1))],
                        [np.zeros((nn,nn)), np.diag(np.ones(nn-1), k=-1) , np.zeros((nn,nn+1))],
                        [np.zeros((nn+1,nn)), np.zeros((nn+1,nn)), np.diag(np.ones(nn), k=-1)]])
        graph[2*nn, nn-1] = 1
        graph[2*nn, 2*nn-1] = 1
        
    elif graph_name =='tent':
        if n_nodes%2 == 0:
            raise ValueError('number of nodes has to be odd for tent')
        hn = int(n_nodes/2)
        graph = np.block([
                         [np.diag(np.ones(hn-1), k=1), np.zeros((hn, hn))],
                         [np.zeros((hn, hn)), np.diag(np.ones(hn-1), k=-1)]
            ])
        graph = np.insert(arr=graph, obj=hn, values=np.zeros(n_nodes-1), axis=0)
        graph = np.insert(arr=graph, obj=hn, values=np.zeros(n_nodes), axis=1)
        graph[hn-1,hn] = 1; graph[hn+1, hn] = 1
        
    elif graph_name == 'ring':
        graph = np.array([[0,0,0,0,1],
                         [1,0,0,0,0],
                         [0,1,0,0,0],
                         [0,0,1,0,0],
                         [0,0,0,1,0]])

    elif graph_name == 'lollipop':
        graph = np.array([[0,0,0,0,0],
                             [1,0,0,0,0],
                             [1,0,0,0,0],
                             [0,1,1,0,0],
                             [0,0,0,1,0]])
        
        
    elif graph_name == 'grid':
        graph = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                         [1,0,0,0,0,0,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0,0,0,0,0,0],
                         [0,0,1,0,0,0,0,0,0,0,0,0],
                         [1,0,0,0,0,0,0,0,0,0,0,0],
                         [0,1,0,0,1,0,0,0,0,0,0,0],
                         [0,0,1,0,0,1,0,0,0,0,0,0],
                         [0,0,0,1,0,0,1,0,0,0,0,0],
                         [0,0,0,0,1,0,0,0,0,0,0,0],
                         [0,0,0,0,0,1,0,0,1,0,0,0],
                         [0,0,0,0,0,0,1,0,0,1,0,0],
                         [0,0,0,0,0,0,0,1,0,0,1,0]])
    return graph

#%% PARAMETERS and DATA
    
# Topology. Canal network
n_nodes = 100
cnm = choose_graph('line', n_nodes=n_nodes)
upstream_bool, downstream_bool = infer_extreme_nodes(cnm) # BC nodes
# BLOCK_LOCATION = downstream_bool.copy()
BLOCK_LOCATION = np.zeros(n_nodes, dtype=bool)

# Physics
B_scale = 7 
B = B_scale * np.ones(n_nodes) # (rectangular) canal width in m
g = 9.8 #* (60*60*24)**2 # m/day
n_manning = 1e-2

# Numerics
ndays = 20
dx = 50
dt = 1 *60*60 # Same units as g
niter = int(ndays/dt)
# For newton method
max_niter_newton = 1000


# GIS data
# for line canal network
bottom = np.linspace(start=4, stop=1,  num=n_nodes)

# for tent canal network:
# bottom = np.hstack((np.linspace(1,3,int(n_nodes/2)), np.linspace(3,1,int(n_nodes/2))))
# bottom = np.insert(bottom, int(n_nodes/2), 3.05)


#%% Vectorial Lax-Friederichs
explicit_laxFriederichs=False
if explicit_laxFriederichs:
    # Compute advection Laplacian
    # L_in_minus, L_in_plus, L_out_minus, L_out_plus = laplacians(cnm)
    L = compute_laplacian_from_adjacency(cnm + cnm.T)
    L_signed = signed_laplacian(cnm)
    
    # Initial and boundary conditions
    # BC
    upstream_bool, downstream_bool = infer_extreme_nodes(cnm) # Get BC nodes
    block_height = 2.2 * BLOCK_LOCATION
    
    # IC
    Y_ini = 1 + max(bottom) - bottom # m above canal bottom
    
    q = 0.0 * np.ones(n_nodes) # Lateral inflow of water
    # q[upstream_bool] = 1
    V_ini = np.zeros(n_nodes)
    V_upstream = 100
    V_ini[upstream_bool] = V_upstream
    # TODO: solve steady state for initial condition of V
    # for i in range(n_nodes-1):
    #     V_ini[i+1] = ((q[i]+q[i+1])/2*dx + V_ini[i]*A_ini[i])/A_ini[i+1]
    
    # sparsify
    # L  = scipy.sparse.csr_matrix(L)
    # L_signed   = scipy.sparse.csr_matrix(L_signed)
    
    V = V_ini.copy()
    Y = Y_ini.copy()
    
    # Simulate. Store solution in terms of Q = AV and Y = A/B
    Q_sol = [0]*niter
    Y_sol = [0]*niter
    for t in range(niter):
        if t % (niter/100) == 0:
            print(f"{100*t/niter} % completed" )
    
        Q_sol[t] = V*B*Y; Y_sol[t] = Y
        
        R = wetted_radius(Y)
        Sf = friction_slope(V, Y)
        F_V = F_of_V(V, Y)
        G_V = G_of_V(V, Y)
        
        Y = Y + lax_friederichs(dx, dt, L, L_signed, Y, V*Y, q/B)
        V = V + lax_friederichs(dx, dt, L, L_signed, V, F_V, G_V)
        
        
        # (Time varying) BC    
        V[upstream_bool] = V_upstream
        V[downstream_bool] = 0.
        # A[upstream_bool] = A_ini[0]
        
        # Downstream BC
    
        # If there is a dam, set limit to height in the dam
        k=1
        V[BLOCK_LOCATION] = 0
        for i,block in enumerate(BLOCK_LOCATION):
            if block == True:
                if Y[i] > block_height[i]:
                    k = 4.
                    V[i] = (k * np.sqrt(g) * (Y[i] - block_height[i])**1.5)/(Y[i]*B[i]) # Eq. 8.101 from Szymkiewicz's book
        

#%% Implicit backwards Euler

@jit(nopython=True)
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

@jit(nopython=True)
def interweave_arrays(array1, array2, shape):
    """
    Takes 2 arrays of the same dimension, [a1, a2, ...] and [b1, b2, ...]
    and returns [a1, b1, a2, b2, ...].
    ai and bi may also be arrays, in which case it would return a matrix
    Shape is the resulting shape of the array. Required parameter by Numba
    """
    result = np.zeros(shape)
    result[0::2] = array1
    result[1::2] = array2
    return result


@jit(nopython=True)
def deinterweave_array(array):
    """
    The inversee of interweave_arrays.
    Takes an interweaved array and returns 2 arrays.
    """
    if array.shape[0] % 2 != 0:
        raise ValueError('the 0th dimension of the array must be even')
    
    return array[0::2], array[1::2]

@jit(nopython=True)
def compute_jV_terms_matrix(dt, A, V, g, n_manning, B):
    bidiag = tridiag(np.ones(2*n_nodes-1), np.ones(2*n_nodes), np.zeros(2*n_nodes-1))
    bidiag[0::2] = np.zeros(2*n_nodes)
    jVjA_term = -4/3*dt*g*n_manning**2*V**2*(2/B + B/A)**(1/3)*B/(A**2)
    jVjV_term = 2*dt*g*n_manning**2*(2/B + B/A)**(4/3)*V
    extra_jV_terms = interweave_arrays(jVjA_term, jVjV_term, 2*jVjA_term.shape[0])
    extra_jV_terms_matrix = np.zeros(shape=(2*n_nodes, 2*n_nodes))
    extra_jV_terms_matrix[1::2] = extra_jV_terms
    extra_jV_terms_matrix
    return np.multiply(extra_jV_terms_matrix, bidiag)

@jit(nopython=True)
def create_jacobian(dt, dx, A, V, g, n_manning, B, A_signed, degree_in_minus_out):

    VA = interweave_arrays(dt/(2*dx) * V, dt/(2*dx) * A, 2*V.shape[0])
    VA2 = interweave_arrays(dt/(2*dx) * g/B, dt/(2*dx) * V, 2*B.shape[0])
    
    # Build sign matrix from incidence matrix
    temp = interweave_arrays(A_signed, A_signed, (2*A_signed.shape[0], A_signed.shape[1]))
    signs = interweave_arrays(temp.T, temp.T, (2*temp.T.shape[0], temp.T.shape[1])).T # interweave by columns by transposing
    
    # diagonal block
    temp = interweave_arrays(degree_in_minus_out, degree_in_minus_out, 
                             (2*degree_in_minus_out.shape[0], degree_in_minus_out.shape[1]))
    signed_diag = interweave_arrays(temp.T, temp.T, (2*temp.T.shape[0], temp.T.shape[1])) # matrix with signs and weights in the diagonal
    
    # jVjA and jVjV extra terms
    extra_jV_terms_matrix = compute_jV_terms_matrix(dt, A, V, g, n_manning, B)
    
    # Finally, construct Jacobian
    jacobian = np.zeros(shape=(2*n_nodes, 2*n_nodes))
    # VA terms
    jacobian[0::2] = VA
    jacobian[1::2] = VA2
    VA_terms_weights = signs + signed_diag
    jacobian = np.multiply(jacobian, VA_terms_weights)
    
    # extra terms in every even row
    jacobian = jacobian + extra_jV_terms_matrix
    
    # -1 in the diagonal
    jacobian = jacobian - np.diag(np.ones(2*n_nodes)) # -1 in the diagonal
    
    return jacobian

@jit(nopython=True)
def F_of_A(A, A_previous, V, q, dt, dx, L_prime):
    return A_previous - A + dt/(2*dx) * L_prime @ (A*V) + dt*q

@jit(nopython=True)
def F_of_V(V, V_previous, A, dt, dx, g, B, bottom, L_prime):
    return V_previous - V + dt/(2*dx)*L_prime @ (0.5*V**2 + g/B*A + g*bottom) + dt*g*friction_slope(V, A/B)

# def adaptive_weight(x, norm_u):
#     # Input is the change in solution x as solved from Newton system
#     # norm_u is the norm of the unknown we are solving
#     w_ceiling = 1e-2
#     w_floor = 1e-3
    
#     weight = w_ceiling * (1 - x/norm_u)
#     return np.max(weight, w_floor)

tstart = time.time()
# Network topology matrices for jacobian
A_signed = cnm - cnm.T
degree_in_minus_out = np.diag(np.sum(A_signed, axis=1)) # incoming - outgoing degree
L_prime = A_signed - degree_in_minus_out

# Initial and boundary conditions
# BC
upstream_bool, downstream_bool = infer_extreme_nodes(cnm) # Get BC nodes
block_height = 2.2 * BLOCK_LOCATION

# IC
Y_ini = 1 + max(bottom) - bottom # m above canal bottom

q = 0.0 * np.ones(n_nodes) # Lateral inflow of water
# q[upstream_bool] = 1
V_ini = 0.001 + np.zeros(n_nodes)
V_upstream = 0
V_ini[upstream_bool] = V_upstream
# TODO: solve steady state for initial condition of V
# for i in range(n_nodes-1):
#     V_ini[i+1] = ((q[i]+q[i+1])/2*dx + V_ini[i]*A_ini[i])/A_ini[i+1]

# sparsify
# L  = scipy.sparse.csr_matrix(L)
# L_signed   = scipy.sparse.csr_matrix(L_signed)

V = V_ini.copy(); V_previous = V_ini.copy()
A = B*Y_ini.copy(); A_previous = B*Y_ini.copy()


# Newton-Rhapson
rel_tol = 1e-3
abs_tol = 1e-3

@jit(nopython=True)
def solve_linear_newtonRaphson(A, A_previous, V, V_previous):
    jacobian = create_jacobian(dt, dx, A, V, g, n_manning, B, A_signed, degree_in_minus_out)
    F_A = F_of_A(A, A_previous, V, q, dt, dx, L_prime)
    F_V = F_of_V(V, V_previous, A, dt, dx, g, B, bottom, L_prime)
    F_u = interweave_arrays(F_A, F_V, 2*F_A.shape[0])
    
    # x = scipy.sparse.linalg.solve(np.asfortranarray(jacobian), -np.asfortranarray(F_u))
    x = np.linalg.solve(np.asfortranarray(jacobian), -np.asfortranarray(F_u))
    
    return x


def newtonRaphson(A, A_previous, V, V_previous):
    for i in range(max_niter_newton):
        x = solve_linear_newtonRaphson(A, A_previous, V, V_previous)
        x_A, x_V = deinterweave_array(x)
        
        weight_A = 1e-3
        weight_V = 1e-3
        print("norm(x_A), norm(A): ", np.linalg.norm(x_A), 0.5*np.linalg.norm(A))
        print("weight_A, weight_V: ",weight_A, weight_V)
        
        A = A + weight_A*x_A; V = V = V + weight_V*x_V
            
        if np.linalg.norm(x) < rel_tol*np.linalg.norm(interweave_arrays(A_previous, V_previous, 2*A_previous.shape[0])) + abs_tol:
            print('\n>>> Newton-Rhapson converged after', i, 'iterations')
            return A, V
        elif np.any(np.isnan(A)):
            print('\n>>> NaN at some point')
            raise ValueError('Nan at some point of Newton-Raphson')
            return 0,0
    return A, V
    
# try:
#     A_sol, V_sol = newtonRaphson(A, A_previous, V, V_previous)
# except:
#     print('An exception occurred in the Newton method')

A_sol = [0]*ndays
V_sol = [0]*ndays
for timestep in tqdm(range(ndays)):
    A, V = newtonRaphson(A, A_previous, V, V_previous)
    A_previous = A
    V_previous = V
    A_sol[timestep] = A
    V_sol[timestep] = V

print(f'\n It took {time.time() - tstart} seconds')

Y_sol = A_sol/B
Q_sol = [V_sol[t] * A_sol[t] for t in range(ndays)]

#%% Plot and animations
plotOpt=True
if plotOpt:
    plt.close('all')
    # Plot initial water height
    niter_to_plot = niter
    
    xx = np.arange(0, dx*n_nodes, dx)
    
    plt.figure()
    plt.fill_between(xx, y1=bottom, y2=0, color='brown', alpha=0.5)
    plt.fill_between(xx, y1=bottom+Y_ini, y2=bottom, color='blue', alpha=0.5)
    plt.title('Initial water height and DEM')
    
    plt.figure()
    plt.plot(V_ini)
    plt.title('Initial velocity')
    plt.show()
    
    # Plot all Qs
    plt.figure()
    
    plt.title('All Qs')
    for t in range(niter_to_plot):
        plt.plot(xx, Q_sol[t], alpha=1/niter_to_plot, color='red')
    
    # Animate Q
    figQ, axQ = plt.subplots()
    axQ.set_title("Animation water flux Q. t = 0")
    axQ.set_ylim(0,10)
    
    lin, = axQ.plot(xx, Q_sol[0], alpha=1.0, color='red')
    
    def animate_Q(t):
        lin.set_ydata(Q_sol[t])  # update the data.
        return lin,
    
    duration_in_frames = 100
    aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
    aniQ.save('output/Q_full_StVenants.mp4')
    
    
    # Plot all Ys
    plt.figure()
    plt.fill_between(xx, y1=bottom, y2=0, color='brown', alpha=0.5)
    plt.title('All Ys')
    for t in range(niter_to_plot):
        plt.plot(xx, bottom + Y_sol[t], alpha=1/niter_to_plot, color='blue')
    
    # Animate Y
    figY, axY = plt.subplots()
    axY.set_title(" Height of water in canal Y")
    axY.set_ylim(0,10)
    axY.fill_between(xx, y1=bottom, y2=0, color='brown', alpha=0.5)
    # dam
    for i, block in enumerate(BLOCK_LOCATION):
        if block:
            dam = patches.Rectangle(xy=(xx[i], bottom[i]), width=0.5, height=block_height[i], linewidth=1, edgecolor='gray', facecolor='gray')
            axY.add_patch(dam)
    
    lin, = axY.plot(xx, bottom + Y_sol[0], alpha=1.0)
    
    def animate_Y(t):
        lin.set_ydata(bottom + Y_sol[t])  # update the data.
        return lin,
    
    aniY = animation.FuncAnimation(figY, animate_Y, frames=range(0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
    aniY.save('output/Y_full_StVenants.mp4')
    
    plt.show()
        
# %%
