# %%
import pandas as pd
import sys
from networkx.utils.rcm import cuthill_mckee_ordering
import pypardiso
import scipy.sparse.csgraph as SSC
import scipy.sparse.linalg as SPL
from numpy.core.fromnumeric import shape
from matplotlib import lines
from scipy.sparse.linalg import splu, spilu, factorized, cg
import scipy.linalg
import networkx as nx
from urllib.parse import MAX_CACHE_SIZE
import matplotlib.pyplot as plt
import pickle
import rasterio
import geopandas
import multiprocessing as mp
import numpy as np
import scipy.sparse
from tqdm import tqdm

from math_preissmann import build_jacobian, solve_linear_system, solve_sparse_linear_system
import utilities
import math_preissmann
import classes
import preprocess_data
# %%
real_data = False
if real_data:
    graph = preprocess_data.load_graph(load_from_pickled=True)
    component_graphs = utilities.find_graph_components(graph)
else:
    n_nodes = 20
    graph_name = 'line'
    cnm = utilities.choose_graph(graph_name, n_nodes=n_nodes)
    bottom = utilities.choose_bottom(graph_name, n_nodes)
    dem = bottom + 10

    # create graph
    graph = nx.from_numpy_array(cnm.T, create_using=nx.DiGraph)
    for n in range(n_nodes):
        graph.nodes[n]['DEM'] = dem[n]

    component_graphs = utilities.find_graph_components(graph)


# %%
# Physics and numerics
NDAYS = 10
general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=50, a=0.6,
                                          max_niter_newton=int(1e5), max_niter_inexact=int(1e3), ntimesteps=24,
                                          rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-3, weight_Q=1e-3)

df_y = pd.DataFrame(index=graph.nodes)
df_Q = pd.DataFrame(index=graph.nodes)
for g_com in tqdm(component_graphs):
    # channel network description
    block_nodes = []  # numbers refer to node names
    block_heights_from_surface = []  # m from DEM surface
    channel_network = classes.ChannelNetwork(
                    g_com, block_nodes, block_heights_from_surface, block_coeff_k=2.0,
                    y_ini_below_DEM=0.4, Q_ini_value=0.0, q=0.26/len(g_com.nodes()), 
                    n_manning=0.05, y_BC_below_DEM=0.0, Q_BC=0.0, channel_width=5)
    
    df_comp_y, df_comp_Q = math_preissmann.simulate_one_component_several_iter(NDAYS, channel_network, general_params)
    
    df_y = pd.concat([df_y, df_comp_y])
    df_Q = pd.concat([df_Q, df_comp_Q])

if not real_data:    
    df_y.plot()

# %% Export solution to geoJSON
if real_data:
    df_nodes = utilities.create_dataframe_of_solutions_at_nodes(
        graph, y_results)

    gdf = utilities.convert_dataframe_to_geodataframe(
        df_nodes, 'x', 'y', 'epsg:32748')

    col_to_export = 'cwl_result'
    gdf_to_export = gdf[[col_to_export, 'geometry']]
    fn_out_cwl = r"C:\Users\03125327\github\canal_net\output\cwl_high_n.geojson"
    # gdf_to_export.to_file(r"C:\Users\03125327\github\canal_networks\output\cwl_geopackage.gpkg", driver="GPKG") # This is not working and it's annoying! Seems to be a Windows problem.
    gdf_to_export.to_file(fn_out_cwl, driver='GeoJSON')

# %% Plot outcome
plotOpt = True
if not real_data and plotOpt:
    import plotting

    # reformat data
    Y_ini = dem - 0.4
    Q_ini = 0.01 * np.ones(n_nodes)
    y_sol = np.array([y_results[i] for i in range(n_nodes)])
    Q_sol = np.array([Q_results[i] for i in range(n_nodes)])

    niter_to_plot = general_params.ntimesteps
    total_iterations = general_params.ntimesteps

    alfa = min(1/niter_to_plot, 0.1)
    xx = np.arange(0, general_params.dx * n_nodes, general_params.dx)

    plotting.plot_water_height(
        xx, Y_ini-bottom, bottom, title='Initial water height and DEM')
    plotting.plot_water_height(
        xx, y_sol-bottom, bottom, title='Final water height and DEM')
    plotting.plot_velocity(xx, Q_ini)

    # plotting.plot_all_Qs(xx, Q_sol, alfa, niter_to_plot, total_iterations)
    # plotting.plot_Qs_animation(
    #     xx, Q_sol, total_iterations, n_frames=niter_to_plot, filename='output/Q_full_StVenants.mp4')
    # plotting.plot_all_Ys(xx, y_sol-bottom, bottom, alfa,
    #                      total_iterations, niter_to_plot)

    # plotting.plot_Ys_animation(
    #     xx, y_sol-bottom, bottom, block_heights, block_nodes, alfa, total_iterations, n_frames=niter_to_plot, filename='output/Y_full_StVenants.mp4')
    # plotting.plot_conservations(y_sol-bottom, Q_sol, total_iterations)

# %%
