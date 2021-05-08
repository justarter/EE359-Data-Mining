import numpy as np
import pandas as pd

def load_data(filepath):
    nodes = []
    edges = []
    row = 0
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            row += 1
            if row == 1:
                continue
            info = line.strip().split(',')
            if not info:
                break
            nodes.append(int(info[0]))
            nodes.append(int(info[1]))
            w = 1
            edges.append(((int(info[0]), int(info[1])), w))
    nodes.sort()
    nodes = list(set(nodes))
    print("The graph has {} nodes, {} edges".format(len(nodes), len(edges)))
    return nodes, edges

def load_truth(PATH):
    ground_truth = {}
    t = 0
    with open(PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            t += 1
            if t == 1:
                continue
            line = line.strip().split(',')
            ground_truth[int(line[0])] = int(line[1])
    return ground_truth

def first_phase(network, communities, k_i, self_loop, edges_per_node, m):
    best_partition = [[node] for node in network[0]]#first, one node is a community
    sum_in = [0] * len(network[0])
    sum_total = [k_i[node] for node in network[0]]
    for e in network[1]:
        if e[0][0] == e[0][1]:  # record self-loop
            sum_in[e[0][0]] += e[1]
            sum_in[e[0][1]] += e[1]

    while True:
        is_change = False
        for node in network[0]:
            original_community = communities[node]# original commmunity
            best_Q_gain = 0
            best_community = original_community
            best_partition[original_community].remove(node)# remove this node from its community
            best_k_i_in = 0

            for e in edges_per_node[node]:
                if e[0][0] == e[0][1]:# if self-loop, don't count
                    continue
                if (e[0][0] == node and communities[e[0][1]] == original_community) \
                        or (e[0][1] == node and communities[e[0][0]] == original_community):
                    best_k_i_in += e[1]
            #remove node
            sum_in[original_community] -= 2*(best_k_i_in + self_loop[node])
            sum_total[original_community] -= k_i[node]
            communities[node] = -1# set -1 for temp
            # search for all neighbors
            neighbors = []
            for e in edges_per_node[node]:
                if e[0][0] == e[0][1]:
                    continue
                if e[0][0] == node:
                    neighbors.append(e[0][1])
                if e[0][1] == node:
                    neighbors.append(e[0][0])
            check = {}
            for neighbor in neighbors:
                k_i_in = 0
                neighbor_community = communities[neighbor]
                if neighbor_community in check:
                    continue
                check[neighbor_community] = 1
                for e in edges_per_node[node]:
                    if e[0][0] == e[0][1]:
                        continue
                    if (e[0][0] == node and communities[e[0][1]] == neighbor_community) \
                            or (e[0][1] == node and communities[e[0][0]] == neighbor_community):
                        k_i_in += e[1]
                Q_gain = 2 * k_i_in-sum_total[neighbor_community]*k_i[node]/m
                # the best community
                if Q_gain > best_Q_gain:
                    best_community = neighbor_community
                    best_Q_gain = Q_gain
                    best_k_i_in = k_i_in
            # move node into best_community
            best_partition[best_community].append(node)
            communities[node] = best_community
            sum_in[best_community] += 2*(best_k_i_in+self_loop[node])
            sum_total[best_community] += k_i[node]
            if original_community != best_community:
                is_change = True
        #all nodes are local optimal
        if not is_change:
            break
    return best_partition, communities, sum_in, sum_total

def second_phase(network, partition, communities):
    new_nodes = [i for i in range(len(partition))]
    # relabel communities
    new_communities = []
    old2new = {}#old label to new label
    label = 0
    for community in communities:
        if community in old2new:
            new_communities.append(old2new[community])
        else:
            new_communities.append(label)
            old2new[community] = label
            label += 1

    new_edges = {}
    for e in network[1]:
        beg = new_communities[e[0][0]]
        end = new_communities[e[0][1]]
        if (beg, end) not in new_edges:
            new_edges[(beg, end)] = e[1]
        else:
            new_edges[(beg, end)] += e[1]
    new_edges = [(key, val) for key, val in new_edges.items()]
    # still one node for one community
    communities = [node for node in new_nodes]

    k_i = [0] * len(new_nodes)
    self_loop = [0] * len(new_nodes)
    edges_per_node = {}
    for edge in new_edges:
        k_i[edge[0][0]] += edge[1]
        k_i[edge[0][1]] += edge[1]
        if edge[0][0] == edge[0][1]:  # self-loop
            self_loop[edge[0][0]] += edge[1]
        # store each node's edges
        if edge[0][0] not in edges_per_node:
            edges_per_node[edge[0][0]] = [edge]
        else:
            edges_per_node[edge[0][0]].append(edge)
        if edge[0][1] not in edges_per_node:
            edges_per_node[edge[0][1]] = [edge]
        elif edge[0][0] != edge[0][1]:  # if self-loop, don't add edge again
            edges_per_node[edge[0][1]].append(edge)

    return new_nodes, new_edges, communities, k_i, self_loop, edges_per_node

def implement(nodes, edges):
    nodes = nodes
    edges = edges
    m = 0  # total graph weight
    communities = [node for node in nodes]
    final_partition = []
    k_i = [0] * len(nodes)
    self_loop = [0] * len(nodes)
    edges_per_node = {}

    for edge in edges:
        m += edge[1]
        k_i[edge[0][0]] += edge[1]
        k_i[edge[0][1]] += edge[1]
        if edge[0][0] == edge[0][1]:# self-loop
            self_loop[edge[0][0]] += edge[1]
        # store each node's edges
        if edge[0][0] not in edges_per_node:
            edges_per_node[edge[0][0]] = [edge]
        else:
            edges_per_node[edge[0][0]].append(edge)
        if edge[0][1] not in edges_per_node:
            edges_per_node[edge[0][1]] = [edge]
        elif edge[0][0] != edge[0][1]:  # if self-loop, don't add edge again
            edges_per_node[edge[0][1]].append(edge)

    network = (nodes, edges)
    Q_max = 0
    iteration = 0
    while True:
        iteration += 1
        partition, communities, sum_in, sum_total = first_phase(network, communities, k_i, self_loop, edges_per_node, m)
        Q = 0
        for i in range(len(partition)):
            Q += sum_in[i] / (m * 2) - (sum_total[i] / (m * 2)) ** 2
        print("Iteration: {}, Q: {}".format(iteration, Q))
        partition = [part for part in partition if part]
        if not final_partition:
            final_partition = partition
        else:
            # based on partiton, convert to actual partition
            actual_partition = []
            for new_part in partition:
                tmp = []
                for node in new_part:
                    tmp.extend(final_partition[node])
                actual_partition.append(tmp)
            final_partition = actual_partition
        #if in this iteration, modularity doesn't change,then stop
        if np.abs(Q-Q_max) < 1e-4:
            break
        new_nodes, new_edges, communities, k_i, self_loop, edges_per_node = second_phase(network, partition, communities)
        network = (new_nodes, new_edges)
        Q_max = Q
    return final_partition, Q_max

def output(PATH, nodes, final_partition, ground_truth):
    results = np.zeros(len(nodes))  # id:category
    for part in final_partition:
        category = np.zeros(5)
        flag = 0
        for node in part:
            if node in ground_truth:
                category[ground_truth[node]] += 1
                flag = 1
        if flag == 1:
            cate = category.argmax()
        else:
            cate = 0
        for node in part:
            results[node] = cate

    results = results.astype(np.int32)

    frame = {"category": results}
    df = pd.DataFrame(frame)
    df.index.name = 'id'
    df.to_csv(PATH)

DATA_PATH = "../data/edges.csv"
GROUND_TRUTH = "../data/ground_truth.csv"
OUTPUT_PATH = "results.csv"

nodes, edges = load_data(DATA_PATH)
ground_truth = load_truth(GROUND_TRUTH)
partition, Q = implement(nodes, edges)
output(OUTPUT_PATH, nodes, partition, ground_truth)
print("Finally get {} paritions with Q={}".format(len(partition), Q))
