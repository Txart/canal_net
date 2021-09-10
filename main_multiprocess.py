# %% imports
import multiprocessing as mp
from tqdm import tqdm

import utilities
import math_preissmann
import classes
import preprocess_data
# %% Some funcs


def multiprocessing_solution_to_dictionary_of_nodes(results):
    y_results = {}
    Q_results = {}
    for result in results:
        y_sol = result[0]
        Q_sol = result[1]
        y_results = utilities.merge_two_dictionaries(y_results, y_sol)
        Q_results = utilities.merge_two_dictionaries(Q_results, Q_sol)
    return y_results, Q_results


def export_cwl_solution_to_geojson(results, graph, general_params):
    y_results, Q_results = multiprocessing_solution_to_dictionary_of_nodes(
        results)
    # Export solution to georeferenced file
    df_nodes = utilities.create_dataframe_of_solutions_at_nodes(
        graph, y_results)

    gdf = utilities.convert_dataframe_to_geodataframe(
        df_nodes, 'x', 'y', 'epsg:32748')

    col_to_export = 'cwl_result'
    gdf_to_export = gdf[[col_to_export, 'geometry']]
    filename = r"C:\Users\03125327\github\canal_net\output\cwl_mp_n={n}_ntimesteps={}.geojson".format(n=general_params.n_manning, n_timesteps=general_params.ntimesteps)
    # gdf_to_export.to_file(r"C:\Users\03125327\github\canal_networks\output\cwl_geopackage.gpkg", driver="GPKG") # This is not working and it's annoying! Seems to be a Windows problem.
    gdf_to_export.to_file(filename, driver='GeoJSON')

    return 0


if __name__ == '__main__':
    n_processes = 6
    y_results = {}
    Q_results = {}
    graph = preprocess_data.load_graph(load_from_pickled=True)
    component_graphs = utilities.find_graph_components(graph)
    # Physics and numerics
    general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=10, a=0.6, n_manning=0.3,
                                              max_niter_newton=int(1e5), max_niter_inexact=int(1e3), ntimesteps=5, rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-3, weight_Q=1e-3)

    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(math_preissmann.simulate_one_component, tqdm(
            [(general_params, g_com) for g_com in component_graphs]))

    # export solution
    export_cwl_solution_to_geojson(results, graph, general_params)
