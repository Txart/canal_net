import multiprocessing as mp
from tqdm import tqdm

import math_preissmann

def multiple_processes(general_params, long_component_graphs):
    with mp.Pool(processes=4) as pool:
        print('simulating000')
        results = pool.starmap(math_preissmann.simulate_one_component, tqdm(
            [(general_params, g_com) for g_com in long_component_graphs]))
    
    return results[-1]