# %% imports
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


# %% PARAMETERS and DATA
# Topology. Canal network
def compute_channel_network_graph(gpkg_fn, fn_dtm10x10):
    # This takes a long time! (approx. 1hour)
    # 1st, clean from streange junctions, then clean from spy nodes
    graph = preprocess_data.read_lines_and_raster_and_produce_dirty_graph(
        gpkg_fn, fn_dtm10x10)
    graph_clean = preprocess_data.clean_graph_from_strange_junctions(graph)
    graph_clean = preprocess_data.clean_graph_from_spy_nodes(
        graph_clean.copy(), mode='delete_all')
    pickle.dump(graph_clean, open("canal_network_matrix.p", "wb"))

    return graph_clean.copy()


def get_component_graphs(graph):
    component_graphs = [graph.subgraph(c).copy() for c in sorted(
        nx.weakly_connected_components(graph), key=len, reverse=True)]
    # Remove components with fewer than 3 nodes
    return [g for g in component_graphs if len(g.nodes) > 3]


def load_graph(load_from_pickled=True):
    if load_from_pickled:
        graph = pickle.load(open("canal_network_matrix.p", "rb"))
    else:
        # Windows
        gpkg_fn = r"C:\Users\03125327\github\canal_net\qgis\final_lines.gpkg"
        fn_dtm10x10 = r"C:\Users\03125327\Documents\qgis\canal_networks\dtm_10x10.tif"
        # Linux -own computer-
        #gpkg_fn = r"/home/txart/Programming/GitHub/canal_net/qgis/final_lines.gpkg"
        #fn_dtm10x10 = r"/home/txart/Programming/data/dtm_10x10.tif"

        graph = compute_channel_network_graph(gpkg_fn, fn_dtm10x10)

    return graph


# %%
# Loop through all components



def multiple_processes(general_params, component_graphs):
    

    return results[-1]


parallel_processing = True
if parallel_processing:
    if __name__ == '__main__':
        n_processes = 10
        y_results = {}
        Q_results = {}
        graph = load_graph(load_from_pickled=True)
        component_graphs = get_component_graphs(graph)
        # Physics and numerics
        general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=50, a=0.6, n_manning=0.15,
                                                  max_niter_newton=100000, max_niter_inexact=1000, ntimesteps=12, rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-3, weight_Q=1e-3)
        with mp.Pool(processes=10) as pool:
            results = pool.starmap(math_preissmann.simulate_one_component, tqdm(
                [(general_params, g_com) for g_com in component_graphs]))

else:
    y_results = {}
    Q_results = {}
    graph = load_graph(load_from_pickled=True)
    component_graphs = get_component_graphs(graph)
    # Physics and numerics
    general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=50, a=0.6, n_manning=0.15,
                                              max_niter_newton=100000, max_niter_inexact=100, ntimesteps=1, rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-3, weight_Q=1e-3)
    for g_com in tqdm(component_graphs):
        ysol, qsol = math_preissmann.simulate_one_component(
            general_params, g_com)
        y_results = utilities.merge_two_dictionaries(y_results, ysol)
        Q_results = utilities.merge_two_dictionaries(Q_results, qsol)

