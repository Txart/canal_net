#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from tqdm import tqdm
import pickle

import preprocess_data
import rasterio
import utilities
import classes
import math_preissmann
import SS_math_preissmann


#%% Import from blopti_dev
import sys
# insert at 1, 0 is the script path
sys.path.insert(1, r'C:\Users\03125327\github\blopti_dev')
import get_data # from blopti_dev folder

#%% Get weather data using functions in blopti_dev.get_data 

fn_weather_data = Path(r'C:\Users\03125327\github\blopti_dev\data\weather_station_historic_data.xlsx')
weather_df = get_data.read_historic_weather_data(fn_weather_data)

# clean data
weather_df = get_data.clean_weather_data(weather_df)

# Compute ET.
# Convert data to daily
daily_weather_df = get_data.aggregate_weather_to_daily(weather_df)
daily_weather_df = get_data.compute_and_append_ET(daily_weather_df)


# Change units P and ET from mm/day to m/day
# daily_weather_df['P'] = daily_weather_df['P']/1000
# daily_weather_df['ET'] = daily_weather_df['ET']/1000

# %% Get daily mean P - ET
# take last x years = 365*x days in the data and get mean daily and variance values
daily_mean_P = np.mean(np.array(daily_weather_df.P)[-365*1:])
daily_mean_ET = np.mean(np.array(daily_weather_df.ET)[-365*1:])

total_daily_source_per_unit_area = daily_mean_P - daily_mean_ET
print(f'Mean daily litres per square meter per day of water is {total_daily_source_per_unit_area:.4f} mm/m^2/day, or {total_daily_source_per_unit_area/24/3600} l/m^2/s')

# Mean litres of water in the whole system is aerial value * system area
# compute catchment area in metres
with rasterio.open(r"C:\Users\03125327\Documents\qgis\canal_networks\dtm_10x10.tif") as src:
    array = src.read()[0]
    not_nan_elements_of_dem = np.count_nonzero(~np.isnan(array))
    catchment_area = not_nan_elements_of_dem*10*10 # 10m is pixel size
 
total_daily_source = total_daily_source_per_unit_area * catchment_area /24/3600/1000 # m/s    
print(f'Mean source of water in the catchment is {total_daily_source:.4f} mm/s')

# %% Simple channel, test effect of different values of n_manning in the final solution

n_nodes = 20
graph_name = 'line'
cnm = utilities.choose_graph(graph_name, n_nodes=n_nodes)
bottom = utilities.choose_bottom(graph_name, n_nodes)
dem = bottom + 10
# All nodes have the same average lateral inflow
q = 0.26/n_nodes* np.ones(n_nodes)

# create graph
graph = nx.from_numpy_array(cnm.T, create_using=nx.DiGraph)
for n in range(n_nodes):
    graph.nodes[n]['DEM'] = dem[n]
    graph.nodes[n]['q'] = q[n]

NDAYS = 10
general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=50, a=0.6,
                                          max_niter_newton=int(1e5), max_niter_inexact=int(1e3), ntimesteps=24,
                                          rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-2, weight_Q=1e-2)

df_y = pd.DataFrame(index=graph.nodes)
df_Q = pd.DataFrame(index=graph.nodes)

N_MANNING = 0.05

# channel network description
block_nodes = []  # numbers refer to node names
block_heights_from_surface = []  # m from DEM surface
channel_network = classes.ChannelNetwork(
                graph, block_nodes, block_heights_from_surface, block_coeff_k=2.0,
                y_ini_below_DEM=0.4, Q_ini_value=0.0, 
                n_manning=N_MANNING, y_BC_below_DEM=0.0, Q_BC=0.0, channel_width=5)

df_comp_y, df_comp_Q = math_preissmann.simulate_one_component_several_iter(NDAYS, channel_network, general_params)
print('simulating...')

df_y = pd.concat([df_y, df_comp_y])
df_Q = pd.concat([df_Q, df_comp_Q])

df_y.plot(title=f'n_manning = {N_MANNING}')



# %% Get lateral inflow per channel network node info

# Read graph
graph = preprocess_data.load_graph(load_from_pickled=True)

# Read flow accumulation raster data
fn_flow_acc = r'C:\Users\03125327\github\canal_net\qgis\lateral_inflow_50x50.tif'
raster_src = rasterio.open(fn_flow_acc)



# append q data to graph nodes
for n,attributes in graph.nodes(data=True):
    coords = (attributes['x'], attributes['y'])
    graph.nodes[n]['q'] = next(raster_src.sample([coords]))[0]

# Get total q. Divide q by that to get fractional flow accumulation
total_q = np.array(list(nx.get_node_attributes(graph, 'q').values())).sum()

# Update q
# q/total_q is the fraction of the total
# then, q/total_q * total_daily_source is daily q in m/s
FRACTION_OF_WATER_INPUT_ENDS_IN_CHANNELS = 1 # F[0,1], 1 meaning that 100% of the incoming water ends up in the channels
for n,attributes in graph.nodes(data=True):
    graph.nodes[n]['q'] = graph.nodes[n]['q']/total_q * total_daily_source * FRACTION_OF_WATER_INPUT_ENDS_IN_CHANNELS



# %% Steady state   DOESN'T WORK RIGHT NOW
component_graphs = utilities.find_graph_components(graph)

general_params = classes.GlobalParameters(g=9.8, dt=0, dx=50, a=0.6,
                    max_niter_newton=int(1e5), max_niter_inexact=0, ntimesteps=0,
                    rel_tol=1e-5, abs_tol=1e-5, weight_A=1e-3, weight_Q=1e-3)


for g_com in tqdm(component_graphs):
    channel_network = classes.ChannelNetwork(
                    g_com, block_nodes=[], block_heights_from_surface=[], block_coeff_k=0,
                    y_ini_below_DEM=0.4, Q_ini_value=0.0, q=0, 
                    n_manning=0.05, y_BC_below_DEM=0.0, Q_BC=0.0, channel_width=5.0)
    
    y_SS, Q_SS = SS_math_preissmann.SS_computation(channel_network, general_params) 

plt.figure()
plt.plot(dem, color='brown')
plt.plot(y_SS, color='blue')

# %% Unsteady
component_graphs = utilities.find_graph_components(graph)

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
                    y_ini_below_DEM=0.4, Q_ini_value=0.0, 
                    n_manning=0.05, y_BC_below_DEM=0.0, Q_BC=0.0, channel_width=5)
    
    df_comp_y, df_comp_Q = math_preissmann.simulate_one_component_several_iter(NDAYS, channel_network, general_params)
    
    df_y = pd.concat([df_y, df_comp_y])
    df_Q = pd.concat([df_Q, df_comp_Q])

pickle.dump(df_y, open('df_y.p', 'wb'))
pickle.dump(df_Q, open('df_Q.p', 'wb'))

# %% View results

df_y = pickle.load(open('df_y.p', "rb"))
df_Q = pickle.load(open('df_Q.p', "rb"))

# Remove rows that have all NaNs 
df_y.dropna(how="all", inplace=True)
# rename columns to strings
df_y.rename(columns={i:str(i) for i in range(0,11)}, inplace=True)

for node,attr in graph.nodes(data=True):
    df_y.loc[node, 'x'] = attr['x']
    df_y.loc[node, 'y'] = attr['y']
    df_y.loc[node, '10'] = df_y.loc[node, '10'] - attr['DEM'] # Convert y of 10th day to cwl = dem - y
    
gdf = utilities.convert_dataframe_to_geodataframe(
    df_y, 'x', 'y', 'epsg:32748')

col_to_export = '10' # must be a string
gdf_to_export = gdf[[col_to_export, 'geometry']]
fn_out_cwl = r"C:\Users\03125327\github\canal_net\output\cwl_cali_n={}.geojson".format(n_manning)
# gdf_to_export.to_file(r"C:\Users\03125327\github\canal_networks\output\cwl_geopackage.gpkg", driver="GPKG") # This is not working and it's annoying! Seems to be a Windows problem.
gdf_to_export.to_file(fn_out_cwl, driver='GeoJSON')

# %% Calibration of numerical params with real data
# Fixed params: tolerances, max_niter_newton
# parameters to fix: max_niter_inexact, weight_A, weight_Q

import time

compo_graphs = component_graphs
