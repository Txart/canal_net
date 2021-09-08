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
import SS_math_preissmann
import classes
import preprocess_data
import main_mp


# %% PARAMETERS and DATA
# Topology. Canal network
raster_data = False
vector_data = True
if raster_data:

    # read data
    cnm, bottom, c_to_r_list = utilities.read_true_data()
    cnm = cnm.toarray()  # Numba needs numpy arrays

    # transposed for definition in networkx
    g = nx.DiGraph(incoming_graph_data=cnm.T)

elif vector_data:
    gpkg_fn = r"C:\Users\03125327\github\canal_net\qgis\final_lines.gpkg"
    fn_dtm10x10 = r"C:\Users\03125327\Documents\qgis\canal_networks\dtm_10x10.tif"

    #gpkg_fn = r"/home/txart/Programming/GitHub/canal_net/qgis/final_lines.gpkg"
    #fn_dtm10x10 = r"/home/txart/Programming/data/dtm_10x10.tif"

    graph = preprocess_data.read_lines_and_raster_and_produce_dirty_graph(
        gpkg_fn, fn_dtm10x10)

    load_from_pickled_graph = True
    if load_from_pickled_graph:
        graph_clean = pickle.load(open("canal_network_matrix.p", "rb"))

    else:  # This takes a long time! (approx. 1hour)
        # 1st, clean from streange junctions, then clean from spy nodes
        graph_clean = preprocess_data.clean_graph_from_strange_junctions(graph)
        graph_clean = preprocess_data.clean_graph_from_spy_nodes(
            graph_clean.copy(), mode='delete_all')
        pickle.dump(graph_clean, open("canal_network_matrix.p", "wb"))

    graph = graph_clean.copy()

else:
    n_nodes = 10
    graph_name = 'Y'
    cnm = utilities.choose_graph(graph_name, n_nodes=n_nodes)
    bottom = utilities.choose_bottom(graph_name, n_nodes)
    dem = bottom + 3


component_graphs = [graph.subgraph(c).copy() for c in sorted(
    nx.weakly_connected_components(graph), key=len, reverse=True)]
length_components = [len(c) for c in sorted(
    nx.weakly_connected_components(graph), key=len, reverse=True)]

# Physics and numerics
general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=50, a=0.6, n_manning=0.15,
                                          max_niter_newton=100000, max_niter_inexact=100, ntimesteps=1, rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-3, weight_Q=1e-3)


# %%
# Loop through all components
all_nodes_y = {n: None for n in range(len(graph.nodes))}
all_nodes_Q = {n: None for n in range(len(graph.nodes))}

long_component_graphs = [g for g in component_graphs if len(g.nodes) > 3]


parallel_processing = True
if parallel_processing:
    if __name__ == '__main__':
        main_mp.multiple_processes(general_params, long_component_graphs)

else:
    y_results = {}
    q_results = {}
    for g_com in tqdm(long_component_graphs):
        ysol, qsol = math_preissmann.simulate_one_component(
            general_params, g_com)
        y_results = utilities.merge_two_dictionaries(y_results, ysol)
        q_results = utilities.merge_two_dictionaries(q_results, qsol)

sys.exit()

# %% Export solutions to raster file


def write_raster_to_disk_given_template(out_filename, raster, template_raster_filename):

    # src file is needed to output with the same metadata and attributes
    with rasterio.open(template_raster_filename) as src:
        profile = src.profile

    profile.update(nodata=None)  # overrun nodata value given by input raster
    # profile.update(dtype='float32', compress='lzw') # What compression to use
    # instead of 64. To save space, we don't need so much precision.
    # float16 is not supported by GDAL, check: https://github.com/mapbox/rasterio/blob/master/rasterio/dtypes.py
    profile.update(dtype='float32')

    with rasterio.open(out_filename, 'w', **profile) as dst:
        dst.write(raster.astype(dtype='float32'), 1)

    return 0


lines_gpkg = geopandas.read_file(gpkg_fn)
raster_src = rasterio.open(fn_dtm10x10)
raster_array = raster_src.read()[0, :, :]  # This works if it has a single band

dict_nodes_to_coords_and_height = dict(graph_clean.nodes.data(True))

cwl_result = np.zeros(shape=raster_array.shape)
for n_node, y_solution in y_results.items():
    x_coord = dict_nodes_to_coords_and_height[n_node]['x']
    y_coord = dict_nodes_to_coords_and_height[n_node]['y']
    dem_at_node = dict_nodes_to_coords_and_height[n_node]['DEM']
    row, col = raster_src.index(x_coord, y_coord)

    cwl_result[row, col] = y_solution - dem_at_node

out_fname = r"C:\Users\03125327\github\canal_networks\output\cwl_solution.tif"
#write_raster_to_disk_given_template(out_filename=out_fname, raster=cwl_result, template_raster_filename=fn_dtm10x10)
with rasterio.open(fn_dtm10x10) as src:
    profile = src.profile
profile.update(nodata=0.0)
with rasterio.open(out_fname, 'w', **profile) as dst:
    dst.write(cwl_result, 1)


# %% Export solution to geopackage
df_nodes = pd.DataFrame.from_dict(
    dict_nodes_to_coords_and_height, orient='index')
df_nodes = df_nodes.reset_index().rename(columns={'index': 'n_node'})
df_nodes['cwl_result'] = df_nodes['n_node'].map(y_results)

# Convert to geodataframe
projection = 'epsg:32748'
gdf = geopandas.GeoDataFrame(df_nodes, geometry=geopandas.points_from_xy(
    df_nodes.x, df_nodes.y), crs=projection)
gdf_to_export = gdf.drop(['n_node', 'x', 'y', 'DEM'], axis=1)

# gdf_to_export.to_file(r"C:\Users\03125327\github\canal_networks\output\cwl_geopackage.gpkg", driver="GPKG") # This is not working and it's annoying! Seems to be a Windows problem.
gdf_to_export.to_file(
    r"C:\Users\03125327\github\canal_networks\output\cwl_geojson.geojson", driver='GeoJSON')

# %%
sys.exit()
# %% Plot, animations and prints
plotOpt = True
if plotOpt:
    import plotting

    niter_to_plot = general_params.ntimesteps
    total_iterations = general_params.ntimesteps

    alfa = min(1/niter_to_plot, 0.1)
    xx = np.arange(0, general_params.dx *
                   channel_network_params.n_nodes, general_params.dx)

    plotting.plot_water_height(xx, Y_ini-bottom, bottom)
    plotting.plot_velocity(xx, Q_ini)
    plotting.plot_all_Qs(xx, Q_sol, alfa, niter_to_plot, total_iterations)
    plotting.plot_Qs_animation(
        xx, Q_sol, total_iterations, n_frames=niter_to_plot, filename='output/Q_full_StVenants.mp4')
    plotting.plot_all_Ys(xx, y_sol-bottom, bottom, alfa,
                         total_iterations, niter_to_plot)

    plotting.plot_Ys_animation(
        xx, y_sol-bottom, bottom, block_heights, block_nodes, alfa, total_iterations, n_frames=niter_to_plot, filename='output/Y_full_StVenants.mp4')
    plotting.plot_conservations(y_sol-bottom, Q_sol, total_iterations)

# %% playground area


unique_junctions = []
for junc in channel_network_params.junctions:
    if junc not in unique_junctions:
        unique_junctions.append(junc)


# %% LU factorization

L, U = scipy.linalg.lu_factor(jacobian)

z = np.linalg.solve(L, -F_u)
x = np.linalg.solve(U, z)


# %%
LU = spilu(scipy.sparse.csc_matrix(jacobian))
LU.solve(-F_u)

# %%
x = SPL.spsolve(scipy.sparse.csc_matrix(jacobian), -F_u, use_umfpack=True)

spijaco = SPL.spilu(scipy.sparse.csc_matrix(
    jacobian), fill_factor=10, drop_tol=1)


# %% with pseudoinverse
PI = scipy.linalg.pinv(jacobian)
scipy.linalg.solve(PI@jacobian, -PI@F_u)


# %% iterative methods
x = scipy.linalg.lstsq(jacobian, -F_u)

# %% permutations

rcm = SSC.reverse_cuthill_mckee(scipy.sparse.csc_matrix(jacobian))
P = np.eye(jacobian[0].shape[0])[rcm, :]
aaa = P@jacobian

LU = splu(scipy.sparse.csc_matrix(aaa))
x = LU.solve(-F_u[rcm])
# %% PARDISO
x = pypardiso.spsolve(scipy.sparse.csc_matrix(jacobian), -F_u)
pypa = pypardiso.factorized(scipy.sparse.csr_matrix(jacobian))
pypa(-F_u)


# %% New trials

y = Y_ini.copy()
Q = Q_ini.copy()
y_previous = y.copy()
Q_previous = Q.copy()

cnm = nx.adjacency_matrix(g_com).toarray().T
cnm = cnm.astype('float64')


rcm = SSC.reverse_cuthill_mckee(scipy.sparse.csc_matrix(cnm))
reverse_permutation = np.argsort(rcm)

stime = time.time()

pversion = True
if pversion:
    cnm = utilities.permute_row_columns(rcm, cnm)
    y = utilities.permute_vector(rcm, y)
    y_previous = utilities.permute_vector(rcm, y_previous)
    Q = utilities.permute_vector(rcm, Q)
    Q_previous = utilities.permute_vector(rcm, Q_previous)
    B = utilities.permute_vector(rcm, B)
    q = utilities.permute_vector(rcm, q)
    q_previous = utilities.permute_vector(rcm, q)


channel_network_params = classes.ChannelNetworkParameters(
    cnm, block_nodes, block_heights, block_coeff_k)

jacobian = build_jacobian(y, y_previous, Q, Q_previous,
                          B, general_params, channel_network_params)
q_previous = q.copy()
F_u = math_preissmann.build_F(y, y_previous, Q, Q_previous,
                              q, q_previous, B, general_params, channel_network_params)


# = pypardiso.spsolve(scipy.sparse.csc_matrix(jacobian), -F_u)
#LU = spilu(scipy.sparse.csc_matrix(jacobian))
#x = LU.solve(-F_u)
x = scipy.linalg.solve(jacobian, -F_u)


x_Q, x_y = utilities.deinterweave_array(x)


if pversion:
    x_Q = utilities.permute_vector(reverse_permutation, x_Q)
    x_y = utilities.permute_vector(reverse_permutation, x_y)

print(time.time() - stime)
#pypa = pypardiso.factorized(scipy.sparse.csr_matrix(jacobian))
# pypa(-F_u)


# %% Draw networkX
G = nx.path_graph(4, create_using=nx.DiGraph)

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)

plt.show()


# %% downstream nodes that are upstream nodes as well
junctions_with_multiple_up = utilities.find_downstream_or_upstream_junction_nodes(
    channel_network_params.cnm, down_or_up='up')
junctions_with_multiple_down = utilities.find_downstream_or_upstream_junction_nodes(
    channel_network_params.cnm, down_or_up='down')

up_junction_nodes = []
down_junction_nodes = []
for up, down in channel_network_params.junctions:
    up_junction_nodes = up_junction_nodes + up
    down_junction_nodes = down_junction_nodes + down

multiple_up_junction_down_nodes = []
multiple_down_junction_up_nodes = []
for _, down in junctions_with_multiple_up:
    multiple_down_junction_up_nodes = multiple_down_junction_up_nodes + down
for up, _ in junctions_with_multiple_down:
    multiple_up_junction_down_nodes = multiple_up_junction_down_nodes + up

up_and_down_junction_nodes = [
    node for node in up_junction_nodes if node in down_junction_nodes]
X_junction_nodes = [
    up_node for up_node in multiple_down_junction_up_nodes if up_node in multiple_up_junction_down_nodes]


junctions_with_multiple_down


down_junction_and_BC_nodes = []
for _, downs in junctions_with_multiple_down:
    down_junction_and_BC_nodes = down_junction_and_BC_nodes + \
        [n for n in downs if n in channel_network_params.downstream_nodes]

# %%
