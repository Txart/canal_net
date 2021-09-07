# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:31:30 2020

@author: 03125327
"""

import utilities
import time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from pathlib import Path
import scipy.sparse
import warnings
from IPython.display import HTML

import preprocess_data

plotOpt = True
# %%
# Functions


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
    # Boundary values below are conditional on this boolean mask
    end_nodes_bool = np.sum(directed_adj_matrix, axis=0) == 0
    first_nodes_bool = np.sum(directed_adj_matrix, axis=1) == 0
    # in case the summing over the sparse matrix changes the numpy array shape
    end_nodes_bool = np.ravel(end_nodes_bool)
    first_nodes_bool = np.ravel(first_nodes_bool)

    return first_nodes_bool, end_nodes_bool


def compute_laplacian_from_adjacency(adj_matrix):
    if np.any(adj_matrix != adj_matrix.T):
        raise ValueError(
            'the matrix must be symmetric, i.e., must be the adj matrix of an undirected graph')
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix

    return laplacian


def compute_L_advection(directed_adj_matrix):
    """
    Returns the 'modified Laplacian', i.e., the advection operator
    """
    D_out = np.diag(np.sum(directed_adj_matrix, axis=0))
    return D_out - directed_adj_matrix


def upwind_from_advection_laplacian(L_adv, downstream_nodes_bool):
    L_upwind = L_adv.copy()
    L_upwind[downstream_nodes_bool,
             downstream_nodes_bool] = L_upwind[downstream_nodes_bool, downstream_nodes_bool] + 1
    return L_upwind


def advection_diffusion_operator(dx, L, L_adv, a, b, diri_bc_bool, neumann_bc_upstream, neumann_bc_downstream, neumann_bc_values):

    # Set default BCs: Neumann
    L_BC = L.copy()  # L doesn't change

    L_advBC = L_adv.copy()
    # Beginning nodes get a -1 in the diagonal
    # L_advBC[neumann_bc_upstream, neumann_bc_upstream] = L[neumann_bc_upstream, neumann_bc_upstream]

    # Ending nodes get a +1 in the diagonal
    # L_advBC[neumann_bc_downstream, neumann_bc_downstream] = L[neumann_bc_downstream, neumann_bc_downstream] + 1

    # Construct operator
    L_mix = a/dx**2*(-L_BC) - b/dx*L_advBC

    # Set Diri BCs
    L_mix[diri_bc_bool] = np.zeros(shape=L_mix[0].shape)

    return L_mix


def set_source_BC(source, dx, a, b, diri_bc_bool, neumann_bc_upstream, neumann_bc_downstream, neumann_bc_values):

    source_BC = np.array(source, dtype=float).copy()
    # Set Neumann BC. No-flux as default
    # Beginning nodes get an extra flux*(-a/dx + b)
    source_BC[neumann_bc_upstream] = source_BC[neumann_bc_upstream] + \
        neumann_bc_values[neumann_bc_upstream] * \
        (-a[neumann_bc_upstream]/dx + b[neumann_bc_upstream])

    # Ending nodes get an extra -flux*a/dx
    source_BC[neumann_bc_downstream] = source_BC[neumann_bc_downstream] - \
        neumann_bc_values[neumann_bc_downstream]*a[neumann_bc_downstream]/dx

    # Set Diri BC
    source_BC[diri_bc_bool] = 0.

    return source_BC


def forward_Euler_adv_diff_single_step(h, dt, L_mix, source):
    return h + dt * L_mix @ h + dt*source


def backwards_Euler(h, dt, L_mix, source):

    P = np.eye(N=L_mix.shape[0]) - dt*L_mix
    P_inv = np.linalg.inv(P)

    h = P_inv @ (h + dt*source)

    return h

# %%
# Canal network and DEM data


def choose_graph(graph_name, n_nodes=5):

    if graph_name == 'line':
        graph = np.array([[0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0]])

    if graph_name == 'long-line':
        graph = np.diag(np.ones(n_nodes-1), k=-1)

    elif graph_name == 'ring':
        graph = np.array([[0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0]])

    elif graph_name == 'lollipop':
        graph = np.array([[0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0],
                          [0, 0, 0, 1, 0]])

    elif graph_name == 'long-Y':
        nn = int(n_nodes/3)  # nodes per branch

        graph = np.block([
            [np.diag(np.ones(nn-1), k=-1), np.zeros((nn, nn)),
             np.zeros((nn, nn+1))],
            [np.zeros((nn, nn)), np.diag(np.ones(nn-1), k=-1),
             np.zeros((nn, nn+1))],
            [np.zeros((nn+1, nn)), np.zeros((nn+1, nn)), np.diag(np.ones(nn), k=-1)]])
        graph[2*nn, nn-1] = 1
        graph[2*nn, 2*nn-1] = 1

    elif graph_name == 'tent':
        if n_nodes % 2 == 0:
            raise ValueError('number of nodes has to be odd for tent')
        hn = int(n_nodes/2)
        graph = np.block([
                         [np.diag(np.ones(hn-1), k=1), np.zeros((hn, hn))],
                         [np.zeros((hn, hn)), np.diag(np.ones(hn-1), k=-1)]
                         ])
        graph = np.insert(arr=graph, obj=hn,
                          values=np.zeros(n_nodes-1), axis=0)
        graph = np.insert(arr=graph, obj=hn, values=np.zeros(n_nodes), axis=1)
        graph[hn-1, hn] = 1
        graph[hn+1, hn] = 1

    elif graph_name == 'grid':
        graph = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
    return graph


true_data = False
if not true_data:
    graph_name = 'long-Y'
    cnm = choose_graph(graph_name, n_nodes=8)

    cnm_sim = cnm + cnm.T
    dem = np.arange(cnm.shape[0])[::-1]
    ini_values = dem.copy()

    # In order to use the same sparse type as the big ass true adjacency matrix
    CNM = scipy.sparse.csr_matrix(cnm)

    n_edges = np.sum(CNM)
    n_nodes = CNM.shape[0]

    # Create NetworkX graph
    g = nx.DiGraph(incoming_graph_data=CNM.T)  # transposed for dynamics!

    def initialize_graph_values(g, h_ini, dem_nodes, diri_bc_values, diri_bc_bool, neumann_bc_values, neumann_bc_bool, source):
        nx.set_node_attributes(
            G=g, values={i: value for i, value in enumerate(h_ini)}, name='h_old')
        nx.set_node_attributes(
            G=g, values={i: value for i, value in enumerate(h_ini)}, name='h_new')
        nx.set_node_attributes(
            G=g, values={i: value for i, value in enumerate(dem_nodes)}, name='ele')
        nx.set_node_attributes(
            G=g, values={i: value for i, value in enumerate(diri_bc_values)}, name='diri_bc')
        nx.set_node_attributes(
            G=g, values={i: value for i, value in enumerate(diri_bc_bool)}, name='diri_bool')
        nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(
            neumann_bc_values)}, name='neumann_bc')
        nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(
            neumann_bc_bool)}, name='neumann_bool')
        nx.set_node_attributes(
            G=g, values={i: value for i, value in enumerate(source)}, name='source')

        return 0

    # symmetric matrix, undirected graph. Useful for dynamics
    g_un = nx.Graph(g)

    # Plot
    one_dim = True
    if one_dim:
        pos = {node: pos for node, pos in enumerate(
            zip(np.arange(n_nodes), dem))}
    else:
        pos = {i: ((i % 4), -int(i/4)) for i in range(0, 12)}
    options = {
        "font_size": 20,
        "node_size": 1000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 5,
        "width": 5,
    }
    if graph_name != 'long-Y':
        nx.draw_networkx(g, pos=pos, **options)

    elif graph_name == 'long-Y':
        nx.draw_networkx(g, **options)

elif true_data:
    filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str)

    dem_rst_fn = Path(
        filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
    can_rst_fn = Path(
        filenames_df[filenames_df.Content == 'canal_raster'].Path.values[0])
    # TODO: use peat depth as depth of canals??
    peat_depth_rst_fn = Path(
        filenames_df[filenames_df.Content == 'peat_depth_raster'].Path.values[0])

    # Choose smaller study area
    # E.g., a study area of (0,-1), (0,-1) is the whole domain
    STUDY_AREA = (0, -1), (0, -1)

    can_arr, wtd_old, dem, _, peat_depth_arr, _, _ = preprocess_data.read_preprocess_rasters(
        STUDY_AREA, dem_rst_fn, can_rst_fn, dem_rst_fn, peat_depth_rst_fn, peat_depth_rst_fn, dem_rst_fn, dem_rst_fn)
    labelled_canals = preprocess_data.label_canal_pixels(can_arr, dem)
    CNM, c_to_r_list = preprocess_data.gen_can_matrix_and_label_map(
        labelled_canals, dem)
    dem_nodes = [dem[loc] for loc in c_to_r_list]
    dem_nodes[0] = 3.0  # something strange happens with this node
    dem_nodes = np.array(dem_nodes)


n_edges = np.sum(CNM)
n_nodes = CNM.shape[0]
DIST_BETWEEN_NODES = 100.  # m

g = 9.8  # m/s**2 gravity accel.

# %%
"""
General advection-diffusion PDE:
    du/dt = au'' - bu'+ source
"""
# Set up simulations
L = compute_laplacian_from_adjacency(cnm_sim)
L_adv = L_advection(cnm)
upstream_bool, downstream_bool = infer_extreme_nodes(cnm_sim)

#L_adv = upwind_from_advection_laplacian(L_adv, downstream_bool)

a = 1 * np.ones(n_nodes)
b = 1 * np.ones(n_nodes)
dx = 100
dt = 0.1
niter = 1000

if np.all(b < 0):
    L_adv = -L_adv
elif np.prod(b < 0):
    raise ValueError(
        'Some but not all of bs are negative. Those digarph edge directions should be reversed!')

#ini_values = dem
ini_values = np.ones(n_nodes)
source = np.array([0]*n_nodes)
u = ini_values.copy()

diri_bc_bool = np.array([False]*n_nodes)
diri_bc_bool[0] = True

neumann_bc_bool = np.array([False]*n_nodes)
neumann_bc_bool[-1] = True
neumann_bc_upstream = neumann_bc_bool * upstream_bool
neumann_bc_downstream = neumann_bc_bool * downstream_bool
# General Neumann BC not implemented yet
neumann_bc_values = 0.001*neumann_bc_bool


if np.any(diri_bc_bool * neumann_bc_bool == True):
    raise ValueError(
        'diri and neumann BC applied at the same time in some node')

#L_upwind = upwind_from_advection_laplacian(L_adv, downstream_bool)
L_upwind = L_adv

L_mix = advection_diffusion_operator(
    dx, L, L_upwind, a, b, diri_bc_bool, neumann_bc_upstream, neumann_bc_downstream, neumann_bc_values)
source_BC = set_source_BC(source, dx, a, b, diri_bc_bool,
                          neumann_bc_upstream, neumann_bc_downstream, neumann_bc_values)

# Simulate
u_sol = [0]*niter
for t in range(niter):
    u_sol[t] = u
    u = forward_Euler_adv_diff_single_step(u, dt, L_mix, source_BC)
    u[u > 2.] = 2.  # Limiting for future surface runoff

# %% Plot and anim
# Plot

if one_dim:
    plt.figure()
    for t in range(niter):
        plt.plot(u_sol[t], color='blue', alpha=0.2)

else:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    ax.plot_surface(xs, ys, u_sol[1].reshape(3, 4))
    ax.scatter(xs, ys, u_sol[1].reshape(3, 4), color='orange')

# Animations


if one_dim:
    fig, ax = plt.subplots()
    ax.set_ylim(0, 3)
    line, = ax.plot(u_sol[0], alpha=1.0)

    def animate(t):
        line.set_ydata(u_sol[t])  # update the data.
        return line,

    duration_in_frames = 100
    ani = animation.FuncAnimation(fig, animate, frames=range(
        0, niter, int(niter/duration_in_frames)))

else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 20)
    xs, ys = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))

    plot = [ax.plot_surface(xs, ys, u_sol[0].reshape(3, 4), color='1')]

    def animate(t, u_sol, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(xs, ys, u_sol[t].reshape(3, 4), color='0.75')
        return plot[0],
    ani = animation.FuncAnimation(fig, animate, fargs=(u_sol, plot))


ani.save('output/gen_adv_diff.mp4')
plt.show()

# %%
"""
Full St Venant: Unsteady
"""
# cnm = choose_graph('long-line', n_nodes=30)
cnm = choose_graph('long-Y', n_nodes=30)
cnm_sim = cnm + cnm.T
n_nodes = cnm.shape[0]
L_adv = L_advection(cnm)
upstream_bool, downstream_bool = infer_extreme_nodes(cnm)
# Add downstream bc to pure advection laplacian
L_upwind = upwind_from_advection_laplacian(L_adv, downstream_bool)

# L_adv = L_upwind

dem = np.linspace(start=4, stop=1,  num=n_nodes)
S0 = dem  # Bottom elevation, i.e., DEM

# params
B = 7  # (rectangular) canal width in m
n_manning = 2.
niter = 10000
dx = 50
dt = 0.005

# Q stuff
Q_ini = np.ones(n_nodes)

Q_source = np.zeros(n_nodes)
Q = Q_ini.copy()
Q_diri_bc_bool = np.array([False]*n_nodes)
# Q_diri_bc_bool[0] = True
Q_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
Q_neumann_bc_values = 0*Q_neumann_bc_bool


# A stuff
Y_ini = np.ones(n_nodes)  # m above canal bottom
A_ini = Y_ini * B
A = A_ini.copy()
A_diri_bc_bool = np.array([False]*n_nodes)
# A_diri_bc_bool[0] = True
A_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
A_neumann_bc_values = 0*A_neumann_bc_bool
A_neumann_bc_upstream = A_neumann_bc_bool * upstream_bool
A_neumann_bc_downstream = A_neumann_bc_bool * downstream_bool
#A_diri_bc_bool[-1] = True


# Diri BC
L_adv[A_diri_bc_bool] = np.zeros(n_nodes)


# Simulate
Q_sol = [0]*niter
Y_sol = [0]*niter
for t in range(niter):
    print(t)
    # q = 0.2 * (np.random.random(n_nodes) - 0.5) # random influx of water
    q = 0 * np.ones(n_nodes)  # influx of water
    q[upstream_bool] = 1.0

    Q_sol[t] = Q
    Y_sol[t] = A/B

    R = A*B/(2*A + B**2)
    Sf = n_manning**2 * Q**2/(A**2)/R**(4/3)

    Q = Q + dt*(-L_adv@(Q**2/A) - 9.8*A/B*L_adv@A + (S0 - Sf)*9.8*A)
    A = A + dt*(-L_adv@Q + q)

    # Set limit to height in the dam
    Y_downstream_limit = 2
    if A[-1] > B*Y_downstream_limit:
        A[-1] = B*Y_downstream_limit


# %% Plot and animations
# Plot initial water height
plt.figure()
plt.fill_between(np.arange(n_nodes), y1=S0, y2=0, color='brown', alpha=0.5)
plt.fill_between(np.arange(n_nodes), y1=S0+Y_ini,
                 y2=S0, color='blue', alpha=0.5)
plt.title('Initial water height and DEM')

plt.figure()
plt.plot(Q_ini)
plt.title('Initial flux')
plt.show()

# Plot all Qs
plt.figure()

plt.title('All Qs')
for t in range(niter):
    plt.plot(Q_sol[t], alpha=0.01, color='red')

# Animate Q
figQ, axQ = plt.subplots()
axQ.set_title("Animattion water flux Q. t = 0")
axQ.set_ylim(0, 10)

lin, = axQ.plot(Q_sol[0], alpha=1.0, color='red')


def animate_Q(t):
    lin.set_ydata(Q_sol[t])  # update the data.
    return lin,


duration_in_frames = 100
aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(
    0, niter, int(niter/duration_in_frames)))
aniQ.save('output/Q_full_StVenants.mp4')

# Plot all Ys
plt.figure()
plt.fill_between(np.arange(n_nodes), y1=S0, y2=0, color='brown', alpha=0.5)
plt.title('All Ys')
for t in range(niter):
    plt.plot(S0 + Y_sol[t], alpha=0.01, color='blue')

# Animate Y
figY, axY = plt.subplots()
axY.set_title("Height of water in canal Y")
axY.set_ylim(0, 10)
axY.fill_between(np.arange(n_nodes), y1=S0, y2=0, color='brown', alpha=0.5)

lin, = axY.plot(S0 + Y_sol[0], alpha=1.0)


def animate_Y(t):
    lin.set_ydata(S0 + Y_sol[t])  # update the data.
    return lin,


aniY = animation.FuncAnimation(figY, animate_Y, frames=range(
    0, niter, int(niter/duration_in_frames)))
aniY.save('output/Y_full_StVenants.mp4')

plt.show()
# %%
"""
Full St Venant: Steady state
"""

# %%
"""
Diffusive wave
"""
raise NotImplementedError('This is  outdated!')
cnm = choose_graph('line')
cnm_sim = cnm + cnm.T
n_nodes = cnm.shape[0]
L = compute_laplacian_from_adjacency(cnm_sim)
L_adv = L_advection(cnm)  # Advection is negative

upstream_bool, downstream_bool = infer_extreme_nodes(cnm_sim)

# params
B = 7  # (rectangular) canal width in m
n_manning = 2.
dx = 100
dt = 0.01

# Q stuff
Q_ini = np.arange(1, n_nodes+1, 1)
Q_source = np.zeros(n_nodes)
Q = Q_ini.copy()
Q_diri_bc_bool = np.array([False]*n_nodes)
#Q_diri_bc_bool[-1] = True
Q_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
Q_neumann_bc_values = 0*Q_neumann_bc_bool
Q_neumann_bc_upstream = Q_neumann_bc_bool * upstream_bool
Q_neumann_bc_downstream = Q_neumann_bc_bool * downstream_bool

#L_mix = advection_diffusion_operator(dx, L, L_adv, D, -l, diri_bc_bool, neumann_bc_bool)
Q_L_mix = advection_diffusion_operator(
    dx, L, L_adv, D, -l, Q_diri_bc_bool, Q_neumann_bc_upstream, Q_neumann_bc_downstream, Q_neumann_bc_values)
A_L_mix = advection_diffusion_operator(dx, L, L_adv, np.zeros(n_nodes), np.ones(
    n_nodes), A_diri_bc_bool, A_neumann_bc_upstream, A_neumann_bc_downstream, A_neumann_bc_values)

# A stuff
Y_ini = np.ones(n_nodes)  # m above canal bottom
A_ini = Y_ini * B
A = A_ini.copy()
A_diri_bc_bool = np.array([False]*n_nodes)
A_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
A_neumann_bc_values = 0*A_neumann_bc_bool
A_neumann_bc_upstream = A_neumann_bc_bool * upstream_bool
A_neumann_bc_downstream = A_neumann_bc_bool * downstream_bool
#A_diri_bc_bool[-1] = True
A_source = np.zeros(n_nodes)

# Simulate
Q_sol = [0]*niter
Y_sol = [0]*niter
for t in range(niter):
    print(t)
    D = A**2 * (A*B/(2*A + B**2)**(4/3))/(2*n_manning**2 * B * Q)
    l = 5/3 * Q/A

    Q_source_BC = set_source_BC(Q_source, dx, D, -l, Q_diri_bc_bool,
                                Q_neumann_bc_upstream, Q_neumann_bc_downstream, Q_neumann_bc_values)
    A_source_BC = set_source_BC(A_source, dx, np.zeros(n_nodes), np.ones(
        n_nodes), A_diri_bc_bool, A_neumann_bc_upstream, A_neumann_bc_downstream, A_neumann_bc_values)

    Q_sol[t] = Q
    Y_sol[t] = A/B

    Q = Q + dt*Q_L_mix @ Q
    A = A + dt*A_L_mix@Q

# %% Plot and animate diffusive wave
# Animate Q
figQ, axQ = plt.subplots()
axQ.set_title("Water flux Q")

lin, = axQ.plot(Q_sol[0], alpha=1.0)


def animate_Q(t):
    lin.set_ydata(Q_sol[t])  # update the data.
    return lin,


aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(
    0, niter, int(niter/duration_in_frames)))
aniQ.save('output/Q_diffusive_wave.mp4')

# Animate Y
figY, axY = plt.subplots()
axY.set_title("Height of water in canal Y")
axY.set_ylim(0, 2)

lin, = axY.plot(Y_sol[0], alpha=1.0)


def animate_Y(t):
    lin.set_ydata(Y_sol[t])  # update the data.
    return lin,


aniY = animation.FuncAnimation(figY, animate_Y, frames=range(
    0, niter, int(niter/duration_in_frames)))
aniY.save('output/Y_diffusive_wave.mp4')


# %%
"""
MacCormak method for full St Venants
"""

n_nodes = 20001
# cnm = choose_graph('long-line', n_nodes=n_nodes)
# cnm = choose_graph('long-Y', n_nodes=n_nodes)
cnm = choose_graph('tent', n_nodes=n_nodes)

# time0 = time.time()

cnm_sim = cnm + cnm.T
L_adv = L_advection(cnm)
upstream_bool, downstream_bool = infer_extreme_nodes(cnm)
# Add downstream bc to pure advection laplacian
L_upwind = upwind_from_advection_laplacian(L_adv, downstream_bool)

L_adv = L_upwind

# sparsify matrices
L_adv = scipy.sparse.csr_matrix(L_adv)
positive_L_adv = np.abs(L_adv)
L_adv_t = L_adv.T
positive_L_adv_t = np.abs(L_adv_t)

# time0 = time.time()
# for i in range(10000):
#     # L_adv.dot(dem)
#     L_adv @ dem
# print(time.time() - time0, "seconds")

# params
B = 7  # (rectangular) canal width in m
g = 9.8  # * 62400**2 IN SECONDS!
n_manning = 1
ndays = 200
dx = 1
dt = 1e-2
niter = int(ndays/dt)

# dem = np.linspace(start=2., stop=0,  num=n_nodes) # downward slope
dem = np.hstack((np.linspace(1, 3, int(n_nodes/2)),
                np.linspace(3, 1, int(n_nodes/2))))
dem = np.insert(dem, int(n_nodes/2), 3.05)


S0 = -L_adv @ dem / dx
S0[upstream_bool] = 0

# Lateral inflw
q = 0.0 * np.ones(n_nodes)  # influx of water

# Q stuff
# Q ini cond from steady state St Venants
Q_left_BC = 2
Q_ini = np.zeros(n_nodes)
Q = Q_ini.copy()
Q[upstream_bool] = Q_left_BC
for i in range(n_nodes-1):
    Q[i+1] = (q[i]+q[i+1])/2*dx + Q[i]

Q[downstream_bool] = 0
# Q_diri_bc_bool[0] = True
Q_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
Q_neumann_bc_values = 0*Q_neumann_bc_bool

BLOCK_LOCATION = downstream_bool.copy()

# A stuff
Y_ini = 2*np.ones(n_nodes) + dem  # m above canal bottom
Y = Y_ini.copy()

Y_diri_bc_bool = np.array([False]*n_nodes)
# Y_diri_bc_bool[0] = True
Y_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
Y_neumann_bc_values = 0*Y_neumann_bc_bool
Y_neumann_bc_upstream = Y_neumann_bc_bool * upstream_bool
Y_neumann_bc_downstream = Y_neumann_bc_bool * downstream_bool
#A_diri_bc_bool[-1] = True

# Diri BC
L_adv[Y_diri_bc_bool] = np.zeros(n_nodes)


def maccormak_step_one(dx, dt, L_adv_t, positive_L_adv_t, F, G):
    # # BCs at downstream nodes
    # sum_matrix[[-1,-1]] = 2
    # difference_matrix[[-1,-1]] = 0
    return dt/dx*(L_adv_t @ F) + 0.5*dt*positive_L_adv_t @ G


def maccormak_step_two(dx, dt, L_adv, positive_L_adv, u_star, F_star, G_star):
    return u_star - dt/dx*L_adv @ F_star + 0.5*dt*np.abs(positive_L_adv) @ G_star

# def maccormak_new_direction_one(dx, dt, L_adv, F, G):
#     return


def maccormak_chaudry_predictor(dx, dt, L_adv, F, G):
    return dt/dx*(-L_adv @ F) + dt*G


def maccormak_chaudry_corrector(dx, dt, L_adv_t, u, F_star, G_star):
    return u + dt/dx * (L_adv_t @ F_star) + dt*G_star


def friction_slope(Q, Y):
    return n_manning**2 * Q**2/((Y*B)**2 * wetted_radius(Y)**(4/3))


def wetted_radius(Y):
    return Y*B/(2*Y + B)


def F_of_Q(Q, Y):
    return Q**2/(B*Y) + 0.5*g*B*Y**2


def G_of_Q(Q, Y):
    return g*Y*B*(S0 - friction_slope(Q, Y))


MACCORMAK_CHAUDRY = False

# Simulate
Q_sol = [0]*niter
Y_sol = [0]*niter
for t in range(niter):
    if t % (niter/100) == 0:
        print(f"{100*t/niter} % completed")
    # q = 0.2 * (np.random.random(n_nodes) - 0.5) # random influx of water

    Q_sol[t] = Q
    Y_sol[t] = Y

    R = wetted_radius(Y)
    Sf = friction_slope(Q, Y)

    # Step 1
    F_Q = F_of_Q(Q, Y)
    G_Q = G_of_Q(Q, Y)

    Q_star = Q + maccormak_step_one(dx, dt,
                                    L_adv_t, positive_L_adv_t, F_Q, G_Q)
    Y_star = Y + maccormak_step_one(dx, dt,
                                    L_adv_t, positive_L_adv_t, Q/B, q/B)

    # New BC introduced by forward terms in space:
    Q_star[downstream_bool] = Q[downstream_bool]
    Y_star[downstream_bool] = Y[downstream_bool]

    # Step 2
    F_star_Q = F_of_Q(Q_star, Y_star)
    G_star_Q = G_of_Q(Q_star, Y_star)

    Q = 0.5*(Q + maccormak_step_two(dx, dt, L_adv,
             positive_L_adv, Q_star, F_star_Q, G_star_Q))
    Y = 0.5*(Y + maccormak_step_two(dx, dt, L_adv,
             positive_L_adv, Y_star, Q_star/B, q/B))

    # Upstream BC
    Q[upstream_bool] = Q_left_BC

    # Y_upstream_reservoir = 1.0
    # Y[upstream_bool] = Y_upstream_reservoir + (1 + 3)/(2*g)*(Q[upstream_bool]/(B*Y[upstream_bool]))**2 # Eq. 14-18 in chaudry's book
    # Y[downstream_bool] = 2.

    # Downstream BC

    # If there is a dam, set limit to height in the dam
    block_height = 2.2 * BLOCK_LOCATION
    block_height[-1] = 3.5
    k = 1
    Q[BLOCK_LOCATION] = 0
    for i, block in enumerate(BLOCK_LOCATION):
        if block == True:
            if Y[i] > block_height[i]:
                k = 5.
                # Eq. 8.101 from Szymkiewicz's book
                Q[i] = k * np.sqrt(g) * (Y[i] - block_height[i])**1.5

    # if MACCORMAK_CHAUDRY:

    #     Q_sol[t] = Q; Y_sol[t] = Y

    #     R = wetted_radius(Y)
    #     Sf = friction_slope(Q, Y)

    #     # Predictor
    #     F_Q = F_of_Q(Q, Y)
    #     G_Q = G_of_Q(Q, Y)

    #     Q_star = Q + maccormak_chaudry_predictor(dx, dt, L_adv, F_Q, G_Q)
    #     Y_star = Y + maccormak_chaudry_predictor(dx, dt, L_adv, Q/B, q/B)

    #     Q_star[downstream_bool] = Q[downstream_bool]
    #     Y_star[downstream_bool] = Y[downstream_bool]

    #     # Corrector
    #     F_star_Q = F_of_Q(Q_star, Y_star)
    #     G_star_Q = G_of_Q(Q_star, Y_star)

    #     Q = 0.5*( Q_star + maccormak_chaudry_corrector(dx, dt, L_adv, Q, F_star_Q, G_star_Q))
    #     Y = 0.5 * (Y_star + maccormak_chaudry_corrector(dx, dt, L_adv, Y, Q_star/B, q/B))

    #     Y[upstream_bool] = 1.
    #     Y[downstream_bool] = 2.


print(time.time() - time0, "seconds")

# %% Plot and animations
plt.close('all')
# Plot initial water height
niter_to_plot = niter

xx = np.arange(0, dx*n_nodes, dx)

plt.figure()
plt.fill_between(np.arange(n_nodes), y1=dem, y2=0, color='brown', alpha=0.5)
plt.fill_between(np.arange(n_nodes), y1=dem+Y_ini,
                 y2=dem, color='blue', alpha=0.5)
plt.title('MacCormak. Initial water height and DEM')

plt.figure()
plt.plot(Q_ini)
plt.title('MacCormak. Initial flux')
plt.show()

# Plot all Qs
plt.figure()

plt.title('MacCormak. All Qs')
for t in range(niter_to_plot):
    plt.plot(Q_sol[t], alpha=1/niter_to_plot, color='red')

# Animate Q
figQ, axQ = plt.subplots()
axQ.set_title("MacCormak. Animattion water flux Q. t = 0")
axQ.set_ylim(0, 10)

lin, = axQ.plot(Q_sol[0], alpha=1.0, color='red')


def animate_Q(t):
    lin.set_ydata(Q_sol[t])  # update the data.
    return lin,


duration_in_frames = 100
aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniQ.save('output/MacCormak_Q_full_StVenants.mp4')


# Plot all Ys
plt.figure()
plt.fill_between(np.arange(n_nodes), y1=dem, y2=0, color='brown', alpha=0.5)
plt.title('MacCormak. All Ys')
for t in range(niter_to_plot):
    plt.plot(dem + Y_sol[t], alpha=1/niter_to_plot, color='blue')

# Animate Y
figY, axY = plt.subplots()
axY.set_title("MacCormak. Height of water in canal Y")
axY.set_ylim(0, 10)
axY.fill_between(np.arange(n_nodes), y1=dem, y2=0, color='brown', alpha=0.5)
# dam
for i, down in enumerate(downstream_bool):
    if down:
        dam = patches.Rectangle(xy=(
            xx[i], dem[i]), width=0.5, height=block_height[i], linewidth=1, edgecolor='gray', facecolor='gray')
        axY.add_patch(dam)

lin, = axY.plot(dem + Y_sol[0], alpha=1.0)


def animate_Y(t):
    lin.set_ydata(dem + Y_sol[t])  # update the data.
    return lin,


aniY = animation.FuncAnimation(figY, animate_Y, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniY.save('output/MacCormak_Y_full_StVenants.mp4')

plt.show()


# %% Neighbour iteration Lax-Friedrichs

n_nodes = 101
# cnm = choose_graph('long-line', n_nodes=n_nodes)
# cnm = choose_graph('long-Y', n_nodes=n_nodes)
cnm = choose_graph('tent', n_nodes=n_nodes)
cnm_sim = cnm + cnm.T

upstream_bool, downstream_bool = infer_extreme_nodes(cnm)

# create dictionaries of neighbours to iterate: predecessors and successors
# write the adj matrix into dictionary form for iteration
edges = scipy.sparse.dok_matrix(cnm)
predecessors = [[] for i in range(0, n_nodes)]
successors = [[] for i in range(0, n_nodes)]

# TODO: information about delta_x should be written in adjacency matrix
for nodes, delta_x in edges.items():
    predecessors[nodes[1]].append((nodes[0], delta_x))
    successors[nodes[0]].append((nodes[1], delta_x))


# params
B = 7  # (rectangular) canal width in m
g = 9.8  # * 62400**2 IN SECONDS!
n_manning = 1
ndays = 200
dx = 1
dt = 1e-2
niter = int(ndays/dt)
BLOCK_LOCATION = downstream_bool.copy()

# dem = np.linspace(start=2., stop=0,  num=n_nodes) # downward slope
bottom = np.hstack((np.linspace(1, 3, int(n_nodes/2)),
                   np.linspace(3, 1, int(n_nodes/2))))
bottom = np.insert(bottom, int(n_nodes/2), 3.05)

# Lateral inflw
q = 0.0 * np.ones(n_nodes)  # influx of water

# V stuff
V_left_BC = 2
V_ini = np.zeros(n_nodes)
V = V_ini.copy()
V[upstream_bool] = V_left_BC
V[downstream_bool] = 0
# Q_diri_bc_bool[0] = True
V_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
V_neumann_bc_values = 0*V_neumann_bc_bool


# A stuff
A_ini = 2*np.ones(n_nodes) - bottom  # m above canal bottom
A = A_ini.copy()

A_diri_bc_bool = np.array([False]*n_nodes)
# Y_diri_bc_bool[0] = True
A_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
A_neumann_bc_values = 0*A_neumann_bc_bool
A_neumann_bc_upstream = A_neumann_bc_bool * upstream_bool
A_neumann_bc_downstream = A_neumann_bc_bool * downstream_bool
#A_diri_bc_bool[-1] = True


def friction_slope(V, A):
    return n_manning**2 * V**2/wetted_radius(A)**(4/3)


def wetted_radius(A):
    return A*B/(2*A + B**2)


def F_of_V(V, A):
    return 0.5*V**2 + g*(A/B + bottom)


def G_of_V(V, A):
    return - g*friction_slope(V, A)


# Simulate
Alist = A.tolist()
Vlist = V.tolist()
Q_sol = [0]*niter
Y_sol = [0]*niter
for t in range(niter):
    if t % (niter/100) == 0:
        print(f"{100*t/niter} % completed")

    Q_sol[t] = V*A
    Y_sol[t] = A/B

    R = wetted_radius(A)
    Sf = friction_slope(V, A)

    F_V = F_of_V(V, A)
    G_V = G_of_V(V, A)
    F_A = V*A

    # Iteration through nodes
    for i in range(0, n_nodes):
        # incoming
        A_incoming = 0
        V_incoming = 0
        if len(predecessors[i]) > 0:
            for pre, w in predecessors[i]:
                A_incoming += A[pre] - A[i] + dt/w*(F_A[pre] + F_A[i])
                V_incoming += V[pre] - V[i] + dt/w*(F_V[pre] + F_V[i])

        # outgoing
        A_outgoing = 0
        V_outgoing = 0
        if len(successors[i]) > 0:
            for suc, w in successors[i]:
                A_outgoing += A[i] - A[suc] + dt/w*(F_A[i] + F_A[suc])
                V_outgoing += V[i] - V[suc] + dt/w*(F_V[i] + F_V[suc])

        Alist[i] = Alist[i] + 0.5*(A_incoming - A_outgoing) + dt*q
        Vlist[i] = Vlist[i] + 0.5*(V_incoming - V_outgoing) - dt*g*Sf

    # Upstream BC
    V[upstream_bool] = V_left_BC

    # Y_upstream_reservoir = 1.0
    # Y[upstream_bool] = Y_upstream_reservoir + (1 + 3)/(2*g)*(Q[upstream_bool]/(B*Y[upstream_bool]))**2 # Eq. 14-18 in chaudry's book
    # Y[downstream_bool] = 2.

    # Downstream BC

    # If there is a dam, set limit to height in the dam
    # block_height = 2.2 * BLOCK_LOCATION
    # block_height[-1] = 3.5
    # k=1
    # Q[BLOCK_LOCATION] = 0
    # for i,block in enumerate(BLOCK_LOCATION):
    #     if block == True:
    #         if Y[i] > block_height[i]:
    #             k = 5.
    #             Q[i] = k * np.sqrt(g) * (Y[i] - block_height[i])**1.5 # Eq. 8.101 from Szymkiewicz's book


# %% Plot and animations
plt.close('all')
# Plot initial water height
niter_to_plot = niter

xx = np.arange(0, dx*n_nodes, dx)

plt.figure()
plt.fill_between(np.arange(n_nodes), y1=dem, y2=0, color='brown', alpha=0.5)
plt.fill_between(np.arange(n_nodes), y1=dem+Y_ini,
                 y2=dem, color='blue', alpha=0.5)
plt.title('MacCormak. Initial water height and DEM')

plt.figure()
plt.plot(Q_ini)
plt.title('MacCormak. Initial flux')
plt.show()

# Plot all Qs
plt.figure()

plt.title('MacCormak. All Qs')
for t in range(niter_to_plot):
    plt.plot(Q_sol[t], alpha=1/niter_to_plot, color='red')

# Animate Q
figQ, axQ = plt.subplots()
axQ.set_title("MacCormak. Animattion water flux Q. t = 0")
axQ.set_ylim(0, 10)

lin, = axQ.plot(Q_sol[0], alpha=1.0, color='red')


def animate_Q(t):
    lin.set_ydata(Q_sol[t])  # update the data.
    return lin,


duration_in_frames = 100
aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniQ.save('output/MacCormak_Q_full_StVenants.mp4')


# Plot all Ys
plt.figure()
plt.fill_between(np.arange(n_nodes), y1=dem, y2=0, color='brown', alpha=0.5)
plt.title('MacCormak. All Ys')
for t in range(niter_to_plot):
    plt.plot(dem + Y_sol[t], alpha=1/niter_to_plot, color='blue')

# Animate Y
figY, axY = plt.subplots()
axY.set_title("MacCormak. Height of water in canal Y")
axY.set_ylim(0, 10)
axY.fill_between(np.arange(n_nodes), y1=dem, y2=0, color='brown', alpha=0.5)
# dam
for i, down in enumerate(downstream_bool):
    if down:
        dam = patches.Rectangle(xy=(
            xx[i], dem[i]), width=0.5, height=block_height[i], linewidth=1, edgecolor='gray', facecolor='gray')
        axY.add_patch(dam)

lin, = axY.plot(dem + Y_sol[0], alpha=1.0)


def animate_Y(t):
    lin.set_ydata(dem + Y_sol[t])  # update the data.
    return lin,


aniY = animation.FuncAnimation(figY, animate_Y, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniY.save('output/MacCormak_Y_full_StVenants.mp4')

plt.show()

# %% Vectorial Lax-Friederichs

n_nodes = 11
cnm = choose_graph('long-line', n_nodes=n_nodes)
# cnm = choose_graph('long-Y', n_nodes=n_nodes)
# cnm = choose_graph('tent', n_nodes=n_nodes)

# time0 = time.time()

cnm_sim = cnm + cnm.T
L_adv = L_advection(cnm)
upstream_bool, downstream_bool = infer_extreme_nodes(cnm)
# Add downstream bc to pure advection laplacian
L_upwind = upwind_from_advection_laplacian(L_adv, downstream_bool)

# L_adv = L_upwind

# sparsify matrices
L_adv = scipy.sparse.csr_matrix(L_adv)
positive_L_adv = np.abs(L_adv)
L_adv_t = L_adv.T
positive_L_adv_t = np.abs(L_adv_t)


# params
B = 7  # (rectangular) canal width in m
g = 9.8  # * 62400**2 IN SECONDS!
n_manning = 1
ndays = 200
dx = 50
dt = 1
niter = int(ndays/dt)

# dem = np.linspace(start=2., stop=0,  num=n_nodes) # downward slope
bottom = np.hstack((np.linspace(1, 3, int(n_nodes/2)),
                   np.linspace(3, 1, int(n_nodes/2))))
bottom = np.insert(bottom, int(n_nodes/2), 3.05)


S0 = -L_adv @ bottom / dx
S0[upstream_bool] = 0

# Lateral inflw
q = 0.0 * np.ones(n_nodes)  # influx of water

# Q stuff
# Q ini cond from steady state St Venants
Q_left_BC = 2
Q_ini = np.zeros(n_nodes)
Q = Q_ini.copy()
Q[upstream_bool] = Q_left_BC
for i in range(n_nodes-1):
    Q[i+1] = (q[i]+q[i+1])/2*dx + Q[i]

Q[downstream_bool] = 0
# Q_diri_bc_bool[0] = True
Q_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
Q_neumann_bc_values = 0*Q_neumann_bc_bool

BLOCK_LOCATION = downstream_bool.copy()

# A stuff
Y_ini = 2*np.ones(n_nodes) + bottom  # m above canal bottom
Y = Y_ini.copy()

Y_diri_bc_bool = np.array([False]*n_nodes)
# Y_diri_bc_bool[0] = True
Y_neumann_bc_bool = np.array([False]*n_nodes)
# General Neumann BC not implemented yet
Y_neumann_bc_values = 0*Y_neumann_bc_bool
Y_neumann_bc_upstream = Y_neumann_bc_bool * upstream_bool
Y_neumann_bc_downstream = Y_neumann_bc_bool * downstream_bool
#A_diri_bc_bool[-1] = True

# Diri BC
L_adv[Y_diri_bc_bool] = np.zeros(n_nodes)

# Translation to A and V
A = Y*B
V = Q/A


def lax_friederichs(dx, dt, L_adv, L_adv_t, positive_L_adv, positive_L_adv_t, u, F, G):
    return 0.5*(-(L_adv + L_adv_t) @ u + dt/dx*(positive_L_adv - positive_L_adv_t) @ F) + dt*G


def friction_slope(V, A):
    return n_manning**2 * V**2/wetted_radius(A)**(4/3)


def wetted_radius(A):
    return A*B/(2*A + B**2)


def F_of_V(V, A):
    return 0.5*V**2 + g*(A/B + bottom)


def G_of_V(V, A):
    return - g*friction_slope(V, A)


# Simulate
Q_sol = [0]*niter
Y_sol = [0]*niter
for t in range(niter):
    if t % (niter/100) == 0:
        print(f"{100*t/niter} % completed")
    # q = 0.2 * (np.random.random(n_nodes) - 0.5) # random influx of water

    Q_sol[t] = V*A
    Y_sol[t] = A/B

    R = wetted_radius(Y)
    Sf = friction_slope(Q, Y)

    F_V = F_of_V(V, A)
    G_V = G_of_V(V, A)

    V = V + lax_friederichs(dx, dt, L_adv, L_adv_t,
                            positive_L_adv, positive_L_adv_t, V, F_V, G_V)
    A = A + lax_friederichs(dx, dt, L_adv, L_adv_t,
                            positive_L_adv, positive_L_adv_t, A, V*A, q)

    # Upstream BC
    Q[upstream_bool] = Q_left_BC

    # Y_upstream_reservoir = 1.0
    # Y[upstream_bool] = Y_upstream_reservoir + (1 + 3)/(2*g)*(Q[upstream_bool]/(B*Y[upstream_bool]))**2 # Eq. 14-18 in chaudry's book
    # Y[downstream_bool] = 2.

    # Downstream BC

    # If there is a dam, set limit to height in the dam
    block_height = 2.2 * BLOCK_LOCATION
    block_height[-1] = 3.5
    k = 1
    V[BLOCK_LOCATION] = 0
    for i, block in enumerate(BLOCK_LOCATION):
        if block == True:
            if A[i] > block_height[i]*B:
                k = 5.
                # Eq. 8.101 from Szymkiewicz's book
                V[i] = A[i]*(k * np.sqrt(g) * (A[i]/B - block_height[i])**1.5)


# %% Plot and animations
plt.close('all')
# Plot initial water height
niter_to_plot = niter

xx = np.arange(0, dx*n_nodes, dx)

plt.figure()
plt.fill_between(np.arange(n_nodes), y1=bottom, y2=0, color='brown', alpha=0.5)
plt.fill_between(np.arange(n_nodes), y1=bottom+Y_ini,
                 y2=bottom, color='blue', alpha=0.5)
plt.title('Lax-Friedrichs. Initial water height and DEM')

plt.figure()
plt.plot(Q_ini)
plt.title('Lax-Friedrichs. Initial flux')
plt.show()

# Plot all Qs
plt.figure()

plt.title('Lax-Friedrichs. All Qs')
for t in range(niter_to_plot):
    plt.plot(Q_sol[t], alpha=1/niter_to_plot, color='red')

# Animate Q
figQ, axQ = plt.subplots()
axQ.set_title("Lax-Friedrichs. Animation water flux Q. t = 0")
axQ.set_ylim(0, 10)

lin, = axQ.plot(Q_sol[0], alpha=1.0, color='red')


def animate_Q(t):
    lin.set_ydata(Q_sol[t])  # update the data.
    return lin,


duration_in_frames = 100
aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniQ.save('output/Lax-Friedrichs_Q_full_StVenants.mp4')


# Plot all Ys
plt.figure()
plt.fill_between(np.arange(n_nodes), y1=bottom, y2=0, color='brown', alpha=0.5)
plt.title('Lax-Friedrichs. All Ys')
for t in range(niter_to_plot):
    plt.plot(bottom + Y_sol[t], alpha=1/niter_to_plot, color='blue')

# Animate Y
figY, axY = plt.subplots()
axY.set_title("Lax-Friedrichs. Height of water in canal Y")
axY.set_ylim(0, 10)
axY.fill_between(np.arange(n_nodes), y1=bottom, y2=0, color='brown', alpha=0.5)
# dam
for i, down in enumerate(downstream_bool):
    if down:
        dam = patches.Rectangle(xy=(xx[i], bottom[i]), width=0.5,
                                height=block_height[i], linewidth=1, edgecolor='gray', facecolor='gray')
        axY.add_patch(dam)

lin, = axY.plot(bottom + Y_sol[0], alpha=1.0)


def animate_Y(t):
    lin.set_ydata(bottom + Y_sol[t])  # update the data.
    return lin,


aniY = animation.FuncAnimation(figY, animate_Y, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniY.save('output/Lax-Friedrichs_Y_full_StVenants.mp4')

plt.show()


# %% Main algorithm. From DEM to numerics


filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str)
dem_rst_fn = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
can_rst_fn = Path(
    filenames_df[filenames_df.Content == 'canal_raster'].Path.values[0])
blocks_fn = Path(filenames_df[filenames_df.Content ==
                 'canal_blocks_raster'].Path.values[0])
sensor_loc_fn = Path(
    filenames_df[filenames_df.Content == 'sensor_locations'].Path.values[0])

can_arr, wtd_old, dem, peat_type_arr, peat_depth_arr, blocks_arr, sensor_loc_arr = preprocess_data.read_preprocess_rasters(
    ((0, -1), (0, -1)), can_rst_fn, can_rst_fn, dem_rst_fn, dem_rst_fn, dem_rst_fn, blocks_fn, sensor_loc_fn)

if 'CNM' and 'labelled_canals' and 'c_to_r_list' not in globals():
    labelled_canals = preprocess_data.label_canal_pixels(can_arr, dem)
    CNM, c_to_r_list = preprocess_data.gen_can_matrix_and_label_map(
        labelled_canals, dem)

built_block_positions = utilities.get_already_built_block_positions(
    blocks_arr, labelled_canals)

n_nodes = CNM.shape[0]

L_adv = L_advection(CNM)
upstream_bool, downstream_bool = infer_extreme_nodes(CNM)
# Add downstream bc to pure advection laplacian
L_upwind = upwind_from_advection_laplacian(L_adv, downstream_bool)

# %% Implicit Backwards Euler

# Compute advection Laplacian
# L_in_minus, L_in_plus, L_out_minus, L_out_plus = laplacians(cnm)
L = compute_laplacian_from_adjacency(cnm + cnm.T)
L_signed = signed_laplacian(cnm)

# Initial and boundary conditions
# BC
upstream_bool, downstream_bool = infer_extreme_nodes(cnm)  # Get BC nodes
block_height = 2.2 * BLOCK_LOCATION

# IC
Y_ini = 1 + max(bottom) - bottom  # m above canal bottom

q = 0.0 * np.ones(n_nodes)  # Lateral inflow of water
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
A = B*Y_ini.copy()

# Construct Jacobian


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


VA = np.zeros((2*n_nodes,))
VA2 = np.zeros((2*n_nodes,))
VA[0::2] = dt/(2*dx) * V
VA[1::2] = dt/(2*dx) * A
VA2[0::2] = dt/(2*dx) * g/B
VA2[1::2] = dt/(2*dx) * V

# Build sign matrix from incidence matrix
A_signed = cnm - cnm.T
temp = np.zeros(shape=(2*n_nodes, n_nodes))
temp[0::2] = A_signed  # put  each row twice
temp[1::2] = A_signed
signs = np.zeros(shape=(2*n_nodes, 2*n_nodes))
signs[:, 0::2] = temp  # put each column twice
signs[:, 1::2] = temp

# diagonal block
temp = np.zeros(shape=(2*n_nodes, n_nodes))
# incoming - outgoing degree
degree_in_minus_out = np.diag(np.sum(A_signed, axis=1))
temp[0::2] = degree_in_minus_out
temp[1::2] = degree_in_minus_out
# matrix with signs and weights in the diagonal
signed_diag = np.zeros(shape=(2*n_nodes, 2*n_nodes))
signed_diag[:, 0::2] = temp
signed_diag[:, 1::2] = temp

# jVjA and jVjV extra terms
bidiag = tridiag(np.ones(2*n_nodes-1), np.ones(2*n_nodes),
                 np.zeros(2*n_nodes-1))
bidiag[0::2] = np.zeros(2*n_nodes)
jVjA_term = -4/3*dt*g*n_manning**2*V**2*(2/B + B/A)**(1/3)*B/(A**2)
jVjV_term = 2*dt*g*n_manning**2*(2/B + B/A)**(4/3)*V
extra_jV_terms = np.zeros(shape=(2*n_nodes))
extra_jV_terms[0::2] = jVjA_term
extra_jV_terms[1::2] = jVjV_term
extra_jV_terms_matrix = np.zeros(shape=(2*n_nodes, 2*n_nodes))
extra_jV_terms_matrix[1::2] = extra_jV_terms
extra_jV_terms_matrix = np.multiply(extra_jV_terms_matrix, bidiag)

# Finally, construct Jacobian
jacob = np.zeros(shape=(2*n_nodes, 2*n_nodes))
# VA terms
jacob[0::2] = VA
jacob[1::2] = VA2
VA_terms_weights = signs + signed_diag
jacob = np.multiply(jacob, VA_terms_weights)

# extra terms in every even row
jacob = jacob + extra_jV_terms_matrix

# -1 in the diagonal
jacob = jacob - np.diag(np.ones(2*n_nodes))  # -1 in the diagonal


def friction_slope(V, A):
    return n_manning**2 * V**2/(wetted_radius(A)**(4/3))


def wetted_radius(A):
    return A/(2*A/B + B)


def F_of_V(V, A):
    return 0.5*V**2 + g*(A/B + bottom)


def G_of_V(V, A):
    return - g*friction_slope(V, A)


# Simulate. Store solution in terms of Q = AV and Y = A/B
Q_sol = [0]*niter
Y_sol = [0]*niter
for t in range(niter):
    if t % (niter/100) == 0:
        print(f"{100*t/niter} % completed")

    Q_sol[t] = V*B*Y
    Y_sol[t] = A/B

    R = wetted_radius(A)
    Sf = friction_slope(V, A)
    F_V = F_of_V(V, A)
    G_V = G_of_V(V, A)

    # (Time varying) BC
    V[upstream_bool] = V_upstream
    V[downstream_bool] = 0.
    # A[upstream_bool] = A_ini[0]

    # Downstream BC

    # If there is a dam, set limit to height in the dam
    k = 1
    V[BLOCK_LOCATION] = 0
    for i, block in enumerate(BLOCK_LOCATION):
        if block == True:
            if Y[i] > block_height[i]:
                k = 4.
                V[i] = (k * np.sqrt(g) * (Y[i] - block_height[i])**1.5) / \
                    (Y[i]*B[i])  # Eq. 8.101 from Szymkiewicz's book


# %% Plot and animations
plt.close('all')
# Plot initial water height
niter_to_plot = niter

xx = np.arange(0, dx*n_nodes, dx)

plt.figure()
plt.fill_between(xx, y1=bottom, y2=0, color='brown', alpha=0.5)
plt.fill_between(xx, y1=bottom+Y_ini, y2=bottom, color='blue', alpha=0.5)
plt.title('Lax-Friedrichs. Initial water height and DEM')

plt.figure()
plt.plot(V_ini)
plt.title('Lax-Friedrichs. Initial velocity')
plt.show()

# Plot all Qs
plt.figure()

plt.title('Lax-Friedrichs. All Qs')
for t in range(niter_to_plot):
    plt.plot(xx, Q_sol[t], alpha=1/niter_to_plot, color='red')

# Animate Q
figQ, axQ = plt.subplots()
axQ.set_title("Lax-Friedrichs. Animation water flux Q. t = 0")
axQ.set_ylim(0, 10)

lin, = axQ.plot(xx, Q_sol[0], alpha=1.0, color='red')


def animate_Q(t):
    lin.set_ydata(Q_sol[t])  # update the data.
    return lin,


duration_in_frames = 100
aniQ = animation.FuncAnimation(figQ, animate_Q, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniQ.save('output/Lax-Friedrichs_Q_full_StVenants.mp4')


# Plot all Ys
plt.figure()
plt.fill_between(xx, y1=bottom, y2=0, color='brown', alpha=0.5)
plt.title('Lax-Friedrichs. All Ys')
for t in range(niter_to_plot):
    plt.plot(xx, bottom + Y_sol[t], alpha=1/niter_to_plot, color='blue')

# Animate Y
figY, axY = plt.subplots()
axY.set_title("Lax-Friedrichs. Height of water in canal Y")
axY.set_ylim(0, 10)
axY.fill_between(xx, y1=bottom, y2=0, color='brown', alpha=0.5)
# dam
for i, block in enumerate(BLOCK_LOCATION):
    if block:
        dam = patches.Rectangle(xy=(xx[i], bottom[i]), width=0.5,
                                height=block_height[i], linewidth=1, edgecolor='gray', facecolor='gray')
        axY.add_patch(dam)

lin, = axY.plot(xx, bottom + Y_sol[0], alpha=1.0)


def animate_Y(t):
    lin.set_ydata(bottom + Y_sol[t])  # update the data.
    return lin,


aniY = animation.FuncAnimation(figY, animate_Y, frames=range(
    0, niter_to_plot, int(niter_to_plot/duration_in_frames)))
aniY.save('output/Lax-Friedrichs_Y_full_StVenants.mp4')

plt.show()


# %% check  independent components in the canal network
g = nx.DiGraph(incoming_graph_data=CNM.T)

length_connected_components = [len(i)
                               for i in nx.weakly_connected_components(g)]
# ForestCarbon network: [1, 5740, 10, 55, 12, 1, 54, 1041, 145, 64, 132, 8, 19, 22, 21, 18, 18]
# Winrock network: largest is around 10000
