# %%
import numpy as np
import multiprocessing as mp
import time
import pickle
import networkx as nx
import tqdm
from numba import njit

import preprocess_data


# %%


def f(x):
    # time.sleep(0.0)
    return x**2, 2*x


def graf(g, dx):
    #cg = ClassGeom(g)
    #cg = nx.to_numpy_array(g).astype('float64').T
    cg = nx.adjacency_matrix(g)
    component_graphs = [g.subgraph(c).copy() for c in sorted(
        nx.weakly_connected_components(g), key=len, reverse=True)]

    # dx = classone.dx
    
    return len(component_graphs[dx]), dx

@njit
def numbis(array1:np.ndarray, array2:np.ndarray) -> float:
    # find maximum of product with iterating for loops to slow down compu
    max = 0.0
    for i in array1:
        for j in array2:
            if i*j > max:
                max = i*j
    return max
        
        
class ClassOne:
    def __init__(self, dx) -> None:
        self.dx = dx
        pass

class ClassGeom:
    def __init__(self, graph) -> None:
        self.cnm = nx.to_numpy_array(graph).astype('float64').T
        pass

# %% Load graph data
gpkg_fn = r"C:\Users\03125327\github\canal_networks\qgis\final_lines.gpkg"
fn_dtm10x10 = r"C:\Users\03125327\github\canal_networks\qgis\dtm_10x10.tif"

graph = preprocess_data.read_lines_and_raster_and_produce_dirty_graph(
    gpkg_fn, fn_dtm10x10)

graph_clean = pickle.load(open("canal_network_matrix.p", "rb"))

cl = ClassOne(dx=50)
# %% Run

if __name__ == '__main__':
    with mp.Pool(processes=4) as pool:
        tstart = time.time()
        res = pool.map_async(f, range(100)).get()

        print(f'>>>> map_async took {time.time() - tstart} seconds')
        print(res)

        tstart = time.time()
        res = pool.map(f, range(100))
        print(f'>>>> map took {time.time() - tstart} seconds')
        print(res)

        # tstart = time.time()
        # res = pool.starmap_async(graf, tqdm.tqdm(
        #     [(graph_clean, 50) for _ in range(10)])).get()
        # print(f'starmpa on graf took {time.time() - tstart} seconds')
        # print(res)
        
        
        tstart = time.time()
        res = pool.starmap(numbis, tqdm.tqdm(
            [(np.random.rand(100), np.random.rand(100)) for _ in range(10)]))
        print(f'starmap on numba took {time.time() - tstart} seconds')
        print(res)

#%% Run with individual processes

# if __name__ == '__main__':
#     process_list = []
#     for i in range(4):
#         p =  mp.Process(target= numbis, args = [np.random.rand(100), np.random.rand(100)])
#         p.start()
#         process_list.append(p)

#     for process in process_list:
#         process.join()
        
# print(f'starmap on numba took {time.time() - tstart} seconds')
# print(res)
# %% tqdm trials

# tqdm.tqdm([i for i in range(10)], total=10)
# %%
