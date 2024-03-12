import numpy as np
import pandas as pd
import pickle
import cupy as cp
import igraph as ig
import leidenalg as la
import cudf

from mygraph import cosine_similarity_cpu, cosine_similarity_gpu, gen_graph_gpu2, save_graph_to_pkl_gpu, load_graph_from_pkl_gpu
from cuml import PCA
from cuml.cluster import KMeans
from tqdm import tqdm

def reduce_pca(features, n_components):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    return reduced_features

def leiden_clustering_cpu(features, threshold, resolution):
    g = ig.Graph(directed=False)
    g.add_vertices(features.shape[0])

    chunk_size = 9000
    num_chunks = int(np.ceil(features.shape[0] / chunk_size))

    for i in tqdm(range(num_chunks)):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, features.shape[0])
        for j in range(i, num_chunks):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, features.shape[0])

            similarities = cosine_similarity(features[start_i:end_i], features[start_j:end_j])

            if i == j:
                similarities[np.tril_indices_from(similarities)] = 0

            sources, targets = np.where(similarities > threshold)
            weights = similarities[sources, targets]

            sources += start_i
            targets += start_j

            g.add_edges(list(zip(sources, targets)))
            g.es[-len(weights):]['weight'] = weights

    partition = la.find_partition(g, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution)
    labels = np.array(partition.membership)

    return labels

def leiden_clustering_gpu2(features, threshold, resolution):
# check
    print(features)
    print("start to generate graph")
# generate graph
    g = gen_graph_gpu2(features, threshold)
# check
    print(g)
    print("start to clustering...")
# clustering
    partition = la.find_partition(g, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution)
    labels = np.array(partition.membership)

    return labels

def leiden_clustering_gpu(features, threshold, resolution):
    '''
    @comments:
        start with Louvain clustering, then leiden clustering
    '''
# load the graph
    #num_vertices, num_edges, num_iterations = save_graph_to_pkl_gpu("/work/bioinformatics/s224636/graph.pkl", features_pca_20, 0.7)
# load graph
    g = load_graph_from_pkl_gpu("/work/bioinformatics/s224636/graph.pkl", 164914, 1798846948, 1)
# stage 1: find the maximum modelity
    labels = list(g.keys())

    print(type(labels))
    '''
# init
        vids = cp.array(range(start_i, end_i))
        cids = cp.array(range(start_i, end_i))
        m = len(weights)
# phrase 1: find the maximum modularity
        for vid_i in vids:
            visit_list_ids = cp.where(sources == vid_i)
            tot_list_ids = cp.where(sources != vid_i)
            max_delta_Q = 0.0
            for vid_j in visit_list_ids:
                delta_Q = (cp.sum(weights[vid_j]) - cp.sum(weights[visit_list_ids])*cp.sum(weights[tot_list_ids])) / (2*m)
                if max_delta_Q <= delta_Q:
                    max_delta_Q = delta_Q
                    max_delta_Q_idx = vid_j
# update label
        if j == num_chunks-1:
            label_h = cp.asnumpy(cids)
            label = np.concatenate((label,label_h))
# update weights, how to compute with the equavilent weights?
    '''

# return
    return labels

def kmeans_clustering_gpu(features, num_clusters, random_state):
    kmeans_gpu = KMeans(n_clusters=num_clusters, random_state=random_state)
    kmeans_gpu.fit(features)
    labels = kmeans_gpu.labels_.get()

    return labels
