import numpy as np
import random
import pandas as pd
from joblib import Parallel, delayed
import itertools
from sklearn.metrics.pairwise import cosine_similarity

class RandomWalk_node2vec:
    def __init__(self, G, p, q):
        self.G = G
        self.p = p
        self.q = q

    def compute_alias_probs(self, pre_node, cur_node):
        probs = []
        for next_node in self.G.neighbors(cur_node):
            weight = 1.0
            if pre_node == next_node:
                probs.append(weight/self.p)
            elif self.G.has_edge(pre_node, next_node):
                probs.append(weight)
            else:
                probs.append(weight/self.q)
        probs = np.array(probs)
        probs /= np.sum(probs)
        return probs

    def create_alias_table(self, probs):
        N = len(probs)
        accept, alias = [0]*N, [0]*N
        alias_probs = probs * N
        low, high = [], []
        for i, prob in enumerate(alias_probs):
            if prob > 1.0:
                high.append(i)
            else:
                low.append(i)

        while low and high:
            low_id, high_id = low.pop(), high.pop()
            accept[low_id] = alias_probs[low_id]
            alias[low_id] = high_id
            alias_probs[high_id] = alias_probs[high_id]-(1-alias_probs[low_id])
            if alias_probs[high_id] < 1.0:
                low.append(high_id)
            else:
                high.append(high_id)

        while high:
            high_id = high.pop()
            accept[high_id] = 1.0
        while low:
            low_id = low.pop()
            accept[low_id] = 1.0
        return accept, alias

    def get_alias_edge(self, pre_node, cur_node):
        probs = self.compute_alias_probs(pre_node, cur_node)
        return self.create_alias_table(probs)

    def alias_sample(self, node, alias):
        idx = np.random.randint(0, len(node))
        rand = np.random.random()
        if rand < node[idx]:
            return idx
        else:
            return alias[idx]

    def preprocess(self):
        alias_nodes = {}
        weight = 1.0
        for node in self.G.nodes():
            probs = np.array([weight]*len(list(self.G.neighbors(node))))
            probs /= np.sum(probs)
            alias_nodes[node] = self.create_alias_table(probs)
        alias_edges = {}
        for edge in self.G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def random_walk(self, len_walk, start_node):
        walk = [start_node]
        while len(walk) < len_walk:
            cur_node = walk[-1]
            cur_neighbors = list(self.G.neighbors(cur_node))
            if len(cur_neighbors) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_neighbors[self.alias_sample(self.alias_nodes[cur_node][0], self.alias_nodes[cur_node][1])])
                else:
                    pre_node = walk[-2]
                    cur_edge = (pre_node, cur_node)
                    walk.append(
                        cur_neighbors[self.alias_sample(self.alias_edges[cur_edge][0], self.alias_edges[cur_edge][1])])
            else:
                break
        return walk

    def get_walks(self, nodes, len_walk, num_walk):
        walks = []
        for _ in range(num_walk):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(len_walk, node))
        return walks

    def partition_work(self, num_walk, workers):
        if num_walk % workers == 0:
            return [num_walk // workers] * workers
        else:  # 如果不整除，会用worker+1个
            return [num_walk // workers] * workers + [num_walk % workers]

    def parallel_get_walks(self, len_walk, num_walk, workers, verbose=1):
        nodes = list(self.G.nodes)
        parallel = Parallel(n_jobs=workers, verbose=verbose)
        res = parallel(delayed(self.get_walks)(nodes, len_walk, num) for num in self.partition_work(num_walk, workers))
        return list(itertools.chain(*res))
