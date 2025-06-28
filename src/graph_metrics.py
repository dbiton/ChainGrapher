import random
from typing import Dict, List, Set
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.approximation.clique import max_clique

import sys
sys.setrecursionlimit(100000)

def graph_average_degree(graph):
    try:
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        if num_nodes == 0:
            return 0.0  # Avoid division by zero if there are no nodes
        avg_degree = (2 * num_edges) / num_nodes
        return avg_degree
    except Exception as e:
        print(f"Exception in graph_average_degree: {e}")
        return float('nan')


def graph_max_degree(graph: nx.Graph):
    try:
        return max([val for (_, val) in graph.degree()], default=0)
    except Exception as e:
        print(f"Exception in graph_average_degree: {e}")
        return float('nan')


def graph_cluster_coe(graph):
    try:
        return nx.average_clustering(graph)
    except Exception as e:
        print(f"Exception in graph_cluster_coe: {e}")
        return float('nan')


def graph_greedy_coloring(graph):
    try:
        coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
        return len(set(coloring.values()))
    except Exception as e:
        print(f"Exception in graph_coloring: {e}")
        return float('nan')


def graph_transitivity(graph):
    try:
        return nx.transitivity(graph)
    except Exception as e:
        print(f"Exception in graph_transitivity: {e}")
        return float('nan')


def graph_assortativity(graph):
    try:
        return nx.degree_assortativity_coefficient(graph)
    except Exception as e:
        print(f"Exception in graph_assortativity: {e}")
        return float('nan')


def graph_modularity(G):
    try:
        communities = list(greedy_modularity_communities(G))
        return nx.algorithms.community.modularity(G, communities)
    except Exception as e:
        print(f"Exception in graph_modularity: {e}")
        return float('nan')


def graph_diameter(G):
    try:
        if nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            diameter = max((nx.diameter(G.subgraph(comp)) for comp in nx.connected_components(G)), default=0)
        return diameter
    except Exception as e:
        print(f"Exception in graph_diameter: {e}")
        return float('nan')


def graph_density(graph):
    try:
        return nx.density(graph)
    except Exception as e:
        print(f"Exception in graph_density: {e}")
        return float('nan')


def graph_conflict_percentage(G):
    try:
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        percentage_conflicts = (num_edges / max_possible_edges) * 100 if max_possible_edges > 0 else 0
        return percentage_conflicts
    except Exception as e:
        print(f"Exception in graph_conflict_percentage: {e}")
        return float('nan')


def graph_longest_path_length(G):
    try:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        longest_path_length = 0
        for comp_nodes in components:
            comp_size = len(comp_nodes)
            if comp_size <= longest_path_length:
                break
            sampled_nodes = random.sample(list(comp_nodes), max(1, comp_size // 10))
            for source_node in sampled_nodes:
                seen_nodes = {source_node}
                curr_node = source_node
                path_length = 1
                while True:
                    neighbors = set(G.neighbors(curr_node))
                    unseen_neighbors = list(neighbors.difference(seen_nodes))
                    if len(unseen_neighbors) == 0:
                        break
                    curr_node = random.choice(unseen_neighbors)
                    seen_nodes.add(curr_node)
                    path_length += 1
                longest_path_length = max(path_length, longest_path_length)
        return longest_path_length
    except Exception as e:
        print(f"Exception in graph_longest_path_length: {e}")
        return float('nan')


# instead make a random path!
def graph_largest_connected_component_size(G):
    try:
        components = nx.connected_components(G)
        largest_size = max((len(component) for component in components), default=0)
        return largest_size
    except Exception as e:
        print(f"Exception in graph_largest_connected_component_size: {e}")
        return float('nan')

def graph_clique_approx(graph):
    try:
        return len(nx.algorithms.approximation.clique.max_clique(graph))
    except Exception as e:
        print(f"Exception in graph_clique_number: {e}")
        return float('nan')


def graph_clique(graph):
    try:
        return max(len(c) for c in nx.find_cliques(graph))
    except Exception as e:
        print(f"Exception in graph_clique_number: {e}")
        return float('nan')


def get_vertex_cover_nx_approx(graph: nx.Graph) -> int:
    try:
        res = nx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover(graph)
        return len(res)
    except Exception as e:
        print(f"Exception in get_vertex_cover_nx_approx: {e}")
        return float('nan')


def get_vertex_cover_pop_max_deg_approx(graph: nx.Graph) -> int:
    try:
        # Create a copy of the graph to avoid modifying the original graph
        graph = graph.copy()
        vertex_cover = set()

        # Repeat until there are no more edges in the graph
        while graph.number_of_edges() > 0:
            # Find the vertex with the highest degree
            highest_degree_node = max(graph.degree, key=lambda x: x[1])[0]

            # Add the vertex to the vertex cover
            vertex_cover.add(highest_degree_node)

            # Remove the selected vertex and its edges from the graph
            graph.remove_node(highest_degree_node)

        return len(vertex_cover)
    except Exception as e:
        print(f"Exception in get_vertex_cover_pop_max_deg_approx: {e}")
        return float('nan')


def get_vertex_cover_dummy_approx(graph: nx.Graph) -> int:
    try:
        # Create a copy of the graph to avoid modifying the original graph
        graph = graph.copy()
        vertex_cover = set()
        
        # the average degree of the graph
        d = 2 * len(graph.edges()) / len(graph.nodes())
        
        # Repeat until there are no more edges in the graph
        while graph.number_of_edges() > 0:
            # Find the vertex with the highest degree
            highest_degree_node, max_degree = max(graph.degree(), key=lambda x: x[1])
            if max_degree < d:
                return len(vertex_cover)
            # highest_degree_node = max(graph.degree, key=lambda x: x[1])[0]

            # Add the vertex to the vertex cover
            vertex_cover.add(highest_degree_node)

            # Remove the selected vertex and its edges from the graph
            graph.remove_node(highest_degree_node)

        return len(vertex_cover)
    except Exception as e:
        print(f"Exception in get_vertex_cover_pop_max_deg_approx: {e}")
        return float('nan')


def get_call_metrics(trace: dict) -> Dict[str, float]:
    results = {
    }
    return results

def graph_degeneracy(G: nx.Graph) -> int:
    try:
        return max(nx.core_number(G).values())
    except Exception as e:
        print(f"Exception in get_vertex_cover_pop_max_deg_approx: {e}")
        return float('nan')

def clique_number_heuristic(G: nx.Graph) -> int:
    return len(nx.algorithms.approximation.clique.max_clique(G))

def get_graph_metrics(graph: nx.Graph) -> Dict[str, float]:
    results = {
        "clique_number_approx": graph_clique_approx(graph),
        # "clique_number": graph_clique(graph),                            # O(2^n)
        #"diameter": graph_diameter(graph),                               # O(n(n + m))
        #"vertex_cover_dummy_approx": get_vertex_cover_dummy_approx(graph),  # O(n^2)
        #"cluster_coe": graph_cluster_coe(graph),                         # O(n * d^2)
        #"transitivity": graph_transitivity(graph),                       # O(n * d^2)
        #"modularity": graph_modularity(graph),                           # O(n log^2 n) to O(n^2)
        "degeneracy": graph_degeneracy(graph),
        "greedy_color": graph_greedy_coloring(graph),                    # O(n + m) to O(n^2)
        "longest_path_length_monte_carlo": graph_longest_path_length(graph),  # O(k(n + m))
        #"vertex_cover_pop_max_deg_approx": get_vertex_cover_pop_max_deg_approx(graph),  # O(n log n + m)
        #"assortativity": graph_assortativity(graph),                     # O(n + m)
        "largest_conn_comp": graph_largest_connected_component_size(graph),  # O(n + m)
        #"vertex_cover_nx_approx": get_vertex_cover_nx_approx(graph),     # O(n + m)
        #"max_degree": graph_max_degree(graph),                           # O(n)
        #"degree": graph_average_degree(graph),                           # O(n)
        "density": graph_density(graph)                                  # O(1)
    }
    return results
