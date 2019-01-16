from collections import namedtuple
from functools import wraps
from math import ceil, floor
from random import sample, randint
from scipy.stats import kendalltau
import networkx as nx
import numpy as np
import os


def remove_edges_uniform(graph, alpha):
    edges_to_remove = sample(graph.edges, ceil(graph.number_of_edges() * alpha))
    new_g = graph.copy()

    new_g.remove_edges_from(edges_to_remove)
    return new_g


def remove_nodes_uniform(graph, alpha):
    nodes_to_keep = sample(graph.nodes, floor(graph.number_of_nodes() * (1 - alpha)))

    return graph.subgraph(nodes_to_keep)


def remove_edges_proportional_degree(mygraph, alpha):
    new_g = mygraph.copy()

    deg = nx.degree(mygraph)
    degsum = np.array([deg[x] + deg[y] for x, y in mygraph.edges()])

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='big'))
    edge_ids = np.random.choice(mygraph.number_of_edges(), size=ceil(mygraph.number_of_edges() * alpha),
                                replace=False, p=degsum / degsum.sum())

    edges_to_remove = np.array(mygraph.edges)[edge_ids, :]

    new_g.remove_edges_from(edges_to_remove)
    return new_g


def add_edges_random(graph, alpha):
    edges_to_add = ceil(graph.number_of_edges() * alpha)
    new_G = graph.copy()
    N = new_G.number_of_nodes()
    edgelist = set([tuple(sorted(e)) for e in new_G.edges()])

    cnt = 0
    new_edges = set()

    while cnt < edges_to_add:
        new_edge = tuple(sorted([randint(0, N - 1), randint(0, N - 1)]))
        if new_edge[0] == new_edge[1]:
            continue
        elif not (new_edge in edgelist or new_edge in new_edges):
            new_edges.add(new_edge)
            cnt += 1

    new_G.add_edges_from(new_edges)
    return new_G


def compare_centrality_dicts_correlation(d1, d2, scipy_correlation=kendalltau):
    if set(d1) != set(d2):
        nodes = sorted(set(d1).intersection(set(d2)))
    else:
        nodes = sorted(d1)

    v1 = np.round([d1[x] for x in nodes], 12)
    v2 = np.round([d2[x] for x in nodes], 12)

    return scipy_correlation(v1, v2).correlation


def robustness_calculator_builder(centrality_measure, comparison_function=compare_centrality_dicts_correlation):
    @wraps(centrality_measure)
    def f(g0, g1):
        return compare_centrality_dicts_correlation(centrality_measure(g0), centrality_measure(g1))
    return f


def estimate_robustness(measured_network, error_mechanism, robustness_calculator, iterations=50, return_values=False):
    measured_robustness = np.array([robustness_calculator(measured_network, error_mechanism(measured_network))
                                    for _ in range(iterations)])
    vals = measured_robustness if return_values else None
    return namedtuple("robustness_estimate", "mean, sd values")(measured_robustness.mean(),
                                                                measured_robustness.std(), vals)
