# %% imports
from tqdm import tqdm
import numpy as np
import pickle
import time
import multiprocessing as mp
import os

import utilities
import math_preissmann
import classes

#%% Bayesian tuning of numerical parameters with real data



def time_spent(weights, ntimesteps, channel_networks):
    start_time = time.time()
    general_params = classes.GlobalParameters(g=9.8, dt=int(86400/ntimesteps), dx=50, a=0.6,
                                        max_niter_newton=int(1e5), max_niter_inexact=int(1e3), ntimesteps=ntimesteps,
                                        rel_tol=1e-5, abs_tol=1e-5, weight_A=weights, weight_Q=weights)

    with mp.Pool(processes=NCPU) as pool:
        _ = pool.starmap(math_preissmann.simulate_one_component, tqdm(
            [(general_params, channel_network) for channel_network in channel_networks]))

    return time.time() - start_time

def objective_function(weights,ntimesteps):
    channel_networks = [classes.ChannelNetwork(
                    g_com, block_nodes=[], block_heights_from_surface=[], block_coeff_k=2.0,
                    y_ini_below_DEM=0.4, Q_ini_value=0.0, 
                    n_manning=0.05, y_BC_below_DEM=0.0, Q_BC=0.0, channel_width=5) for g_com in component_graphs]

    return time_spent(weights, ntimesteps, channel_networks)
    
# %% Tuning in parallel
# I want to obtain the best numerical parameters that compute the simulation without crashing
# I know 2 things:
# 1. the bigger dt, the less iterations, which implies less time
# 2. the larger the weights of Newton-Raphson method, the less time
# The solver works for dt=3600 and weights=1e-3. Can we improve this?
# I do an exhaustive grid search with different dts and weights
# Instead of using dt directly, I use ntimesteps (number of chunks in which a day is divided):
# ntimesteps is directly related to dt, because total simulation time is 1day. So dt=int(86400/ntimesteps)


if __name__ == '__main__':

    NCPU = os.cpu_count()
    
    n_components_to_compute = int(NCPU*2)
    
    graph = pickle.load(open("canal_network_matrix_with_q.p", "rb")) # q in the grapgh nodes
    component_graphs = utilities.find_graph_components(graph)[:n_components_to_compute] # Take the largest components
    
    NTIMESTEPS_RANGE  = np.arange(start=2, stop=24, step=2)[::-1]
    WEIGHTS_RANGE = [10**(i) for i in np.linspace(start=-3, stop=-1, num=20)]
    
    results = {}
    
    for ntimesteps in tqdm(NTIMESTEPS_RANGE):
        for weights in WEIGHTS_RANGE:
            try:
                timespent = objective_function(weights, ntimesteps)
                results[(ntimesteps, weights)] = timespent
            except: # the computation crashed and it is not expected to improve by additional refinement of parameters
                break
    
    pickle.dump(results, open('tuning_res.p', 'wb'))



# %%
