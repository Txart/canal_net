# %% imports
from tqdm import tqdm
import pandas as pd
import pickle
import time
import skopt
import joblib

import utilities
import math_preissmann
import classes

#%% Bayesian tuning of numerical parameters with real data

graph = pickle.load(open("canal_network_matrix_with_q.p", "rb")) # q in the grapgh nodes
component_graphs = utilities.find_graph_components(graph)[100:110] # Take 10 medium sized components

def time_spent(weight_A, weight_Q, channel_networks):
    try:
        start_time = time.time()
        general_params = classes.GlobalParameters(g=9.8, dt=3600, dx=50, a=0.6,
                                            max_niter_newton=int(1e5), max_niter_inexact=int(1e3), ntimesteps=24,
                                            rel_tol=1e-5, abs_tol=1e-5, weight_A=weight_A, weight_Q=weight_Q)

        for channel_network in channel_networks:
            _ = math_preissmann.simulate_one_component(general_params, channel_network)

        return time.time() - start_time
    
    except: # solver crashed with some params
        return 10e8

def objective_function(weight_A, weight_Q):
    channel_networks = [classes.ChannelNetwork(
                    g_com, block_nodes=[], block_heights_from_surface=[], block_coeff_k=2.0,
                    y_ini_below_DEM=0.4, Q_ini_value=0.0, 
                    n_manning=0.05, y_BC_below_DEM=0.0, Q_BC=0.0, channel_width=5) for g_com in component_graphs]

    return -1.0 * time_spent(weight_A, weight_Q, channel_networks)
    
# %% In parallel: https://scikit-optimize.github.io/stable/auto_examples/parallel-optimization.html#sphx-glr-auto-examples-parallel-optimization-py
SPACE = [
    skopt.space.Real(0, 0.1, name='weight_y', prior='uniform'),
    skopt.space.Real(0, 0.1, name='weight_Q', prior='uniform')]

optimizer = skopt.Optimizer(dimensions=SPACE)

N_ITER = 10
N_CPU = 6
for i in range(N_ITER):
    print(i)
    x = optimizer.ask(n_points=N_CPU)
    y = joblib.Parallel(n_jobs=N_CPU)(joblib.delayed(objective_function)(*v) for v in x)
    optimizer.tell(x,y)
    

# %% Print results
# optimizer.Xi # list of params used
print(min(optimizer.yi))
