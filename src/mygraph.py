import os
import sys

import numpy as np
import pandas as pd
import pickle
import cupy as cp
import igraph as ig
import leidenalg as la
import cudf
from numba import jit

from sklearn.metrics.pairwise import cosine_similarity
from cuml import PCA
from cuml.cluster import KMeans
from tqdm import tqdm

def cosine_similarity_cpu(X, Y):
    similarities = cosine_similarity(X.get(), Y.get())

    return similarities

def cosine_similarity_gpu(X, Y):
    X_norm = cp.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = cp.linalg.norm(Y, axis=1, keepdims=True)
    similarities = cp.dot(X, Y.T) / cp.dot(X_norm, Y_norm.T)

    return similarities

def gen_graph_gpu2(features, threshold):
    g = ig.Graph(directed=False)
    g.add_vertices(features.shape[0])

    sources_h = np.zeros(1, dtype='int32')
    targets_h = np.zeros(1, dtype='int32')
    weights_h = np.ones( 1, dtype='float32')

    chunk_size = 1000
    group_size = 5

    num_chunks = int(np.ceil(features.shape[0] / chunk_size))
    iters = [(i, j) for i in range(num_chunks) for j in range(i, num_chunks)]
    num_groups = int(len(iters) / group_size)
# compute the graph for clustering, iGraph graph, matching leidenalg
    for k in tqdm(range(num_groups)):
        sources_tmp = cp.zeros(1, dtype='int32')
        targets_tmp = cp.zeros(1, dtype='int32')
        weights_tmp = cp.ones( 1, dtype='float32')
        for i, j in iters[k*group_size:(k+1)*group_size]:
            start_i = i * chunk_size
            end_i = min((i+1) * chunk_size, features.shape[0])
            start_j = j * chunk_size
            end_j = min((j+1) * chunk_size, features.shape[0])

            similarities = cosine_similarity_gpu(features[start_i:end_i], features[start_j:end_j])

            if i == j:
                similarities[cp.tril_indices_from(similarities)] = 0

            sources, targets = cp.where(similarities > threshold)
            weights = similarities[sources, targets]

            sources += start_i
            targets += start_j

            sources_tmp = cp.concatenate((sources_tmp, sources))
            targets_tmp = cp.concatenate((targets_tmp, targets))
            weights_tmp = cp.concatenate((weights_tmp, weights))
# copy arr's back to host
        sources_h = np.concatenate((sources_h, cp.asnumpy(sources_tmp).astype(np.int32)))
        targets_h = np.concatenate((targets_h, cp.asnumpy(targets_tmp).astype(np.int32)))
        weights_h = np.concatenate((weights_h, cp.asnumpy(weights_tmp).astype(np.float32)))

    edges_h = list(zip(sources_h, targets_h)) # where the memory problem lies

    if len(weights_h) != 0:
        g.add_edges(edges_h) # still slow
        g.es['weight'] = weights_h

    return g

@jit(nopython=True)
def gen_graph_from_arrays(sources, targets):
    '''
    @comments:
        use dictionary to store the graph
    '''
    s = []
    l = len(sources)
    i = 0
    trace = 0
    while i < len(targets):
        c = []
        for j in range(i, l):
            if sources[j] != trace:
                break
            c.append(targets[j])
            i = i + 1
        s.append(c)
        trace = trace + 1

    return s

def save_graph_to_pkl_gpu(graph_file, features, threshold):
    '''
    @comments:
       get arrays of sources, targets and weights, use I/O for better performance
    '''
    chunk_size = 8000 # GPU memory limitation
    total_size = features.shape[0]
    num_chunks = int(np.ceil(total_size / chunk_size))
    label = np.array(range(chunk_size))
    iters = [(i, j) for i in range(num_chunks) for j in range(i, num_chunks)]
# check if the file is already exsited, if so, the file will be deleted
    if os.path.exists(graph_file):
        os.remove(graph_file)
        print("File detected and removed!")
# open the file
    print("Start to write the arrays to the file...")
    total_size_weights = 0
    with open(graph_file, 'wb') as file:
# compute the arrays of sources, targets and weights
        for i, j in tqdm(iters):
            start_i = i * chunk_size
            end_i = min((i+1) * chunk_size, features.shape[0])
            start_j = j * chunk_size
            end_j = min((j+1) * chunk_size, features.shape[0])

            similarities = cosine_similarity_gpu(features[start_i:end_i], features[start_j:end_j])

            if i == j:
                similarities[cp.tril_indices_from(similarities)] = 0

            sources, targets = cp.where(similarities > threshold)
            weights = similarities[sources, targets]

            sources += start_i
            targets += start_j

            pickle.dump(sources, file)
            pickle.dump(targets, file)
            pickle.dump(weights, file)

            total_size_weights = total_size_weights + len(weights)

    return total_size, total_size_weights, len(iters)

def load_graph_from_pkl_gpu(graph_file, num_vertices, num_edges, num_iterations):
    '''
    @comments:
        get graph from newly created .pkl file by save_graph_to_pkl_gpu()
    '''
    print(num_vertices)
    print(num_edges)
    print(num_iterations)
# use igraph
    print("Start to load file...")
    g = {}
    c = 0
# load arrays to create graph(dictionary is used here)
    with open(graph_file, 'rb') as file:
        for i in tqdm(range(num_iterations)):
            # copy cupy.ndarray back, since the dump is based on cupy.ndarray
            sources = pickle.load(file).get()
            targets = pickle.load(file).get()
            weights = pickle.load(file).get()
# generate graph
            g_tmp = {i+c : j for i, j in enumerate(zip(sources, targets))}
            g.update(g_tmp)

            c = c + len(weights)

    return g
