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


y_results = {}
Q_results = {}
graph = preprocess_data.load_graph(load_from_pickled=True)
component_graphs = utilities.find_graph_components(graph)
# Physics and numerics
general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=10, a=0.6, n_manning=1,
                                          max_niter_newton=10000, max_niter_inexact=100, ntimesteps=1, rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-3, weight_Q=1e-3)
for g_com in tqdm(component_graphs):
    ysol, qsol = math_preissmann.simulate_one_component(
        general_params, g_com)
    y_results = utilities.merge_two_dictionaries(y_results, ysol)
    Q_results = utilities.merge_two_dictionaries(Q_results, qsol)

# %% Export solution to geoJSON
df_nodes = utilities.create_dataframe_of_solutions_at_nodes(graph, y_results)

gdf = utilities.convert_dataframe_to_geodataframe(df_nodes, 'x', 'y', 'epsg:32748')

col_to_export = 'cwl_result'
gdf_to_export = gdf[[col_to_export, 'geometry']]
fn_out_cwl = r"C:\Users\03125327\github\canal_net\output\cwl_high_n.geojson"
# gdf_to_export.to_file(r"C:\Users\03125327\github\canal_networks\output\cwl_geopackage.gpkg", driver="GPKG") # This is not working and it's annoying! Seems to be a Windows problem.
gdf_to_export.to_file(fn_out_cwl, driver='GeoJSON')

# %%
