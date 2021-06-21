import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import networkx as nx
from collections import Counter
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from dataset import MyDataset
from model import MyWord2Vec
from randomwalk_node2vec import RandomWalk_node2vec

class Node2Vec:
    def __init__(self, G, len_walk, num_walk, workers, p, q):
        self.G = G
        self.len_walk = len_walk
        self.num_walk = num_walk
        self.randomwalk = RandomWalk_node2vec(G=G, p=p, q=q)
        self.randomwalk.preprocess()
        self.sentences = self.randomwalk.parallel_get_walks(len_walk=len_walk, num_walk=num_walk, workers=workers, verbose=1)

    def train(self, embedding_size, window_size, iterations, batch_size, lr, negative):
        text = []
        for s in self.sentences:
            text.extend(s)
        vocab_size = len(np.unique(text))

        ordered_vocab = dict(Counter(text).most_common())
        self.word2idx = {word: i for i, word in enumerate(ordered_vocab.keys())}
        self.idx2word = {i: word for i, word in enumerate(ordered_vocab.keys())}

        word_counts = np.array([count for count in ordered_vocab.values()], dtype=np.float32)
        word_counts = word_counts ** (3. / 4.)  # 论文说用0.75次方
        word_freqs = word_counts / np.sum(word_counts)

        dataset = MyDataset(text, self.word2idx, word_freqs, window_size, negative, self.len_walk)
        dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)

        print("=======>Learning Embedding")

        model = MyWord2Vec(vocab_size, embedding_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            print("=======>Use cuda")
            model.cuda()
        else:
            print("=======>Use cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for e in range(iterations):
            for i, (center_words, positive_words, negative_words) in enumerate(dataloader):
                center_words = center_words.long().to(device)
                positive_words = positive_words.long().to(device)
                negative_words = negative_words.long().to(device)

                optimizer.zero_grad()
                loss = model.forward(center_words, positive_words, negative_words).mean()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print('Epoch', e, 'Iteration', i, "Loss", loss.item())
                if i % 5000 == 0:
                    print("=======>Store Embedding")
                    _embeddings = model.store_embedding()
                    embeddings = {}
                    for i in range(len(_embeddings)):
                        embeddings[self.idx2word[i]] = _embeddings[i]

                    with open('embedding.txt', 'w') as f:
                        for node in embeddings:
                            f.write(str(node) + ' ' + ' '.join(map(str, embeddings[node])) + '\n')
        print("=======>Learing Finished")
        return model

    def data_from_files(self):
        text = []
        with open('walks_len30_num10.txt') as f:
            walks = f.readlines()
            for walk in walks:
                walk = walk.strip().split('\t')
                text.append(list(map(int, walk)))
        self.sentences = text

def test():
    test_data = pd.read_csv('../data/course3_test.csv')
    test_data = np.array(test_data)#10246*3

    embeddings = {}
    with open("embedding.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            embeddings[int(data[0])] = np.array(data[1:]).astype(np.float64)

    res = {}
    for i in range(len(test_data)):
        node1 = test_data[i, 1]
        node2 = test_data[i, 2]
        if node1 not in embeddings or node2 not in embeddings:
            res[test_data[i, 0]] = 0.0000
            continue
        res[test_data[i, 0]] = round(float(cosine_similarity(embeddings[node1].reshape(1,-1), embeddings[node2].reshape(1,-1))), 4)

    df = pd.DataFrame.from_dict(res, orient="index", columns=["label"])
    df.index.name = 'id'
    df.to_csv("submission.csv")



if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    len_walk = 30
    num_walk = 10
    workers = 2
    p = 0.4
    q = 1.0

    embedding_size = 128
    window_size = 5  # context window size
    iterations = 1
    batch_size = 32
    lr = 1e-3
    negative = 4  # number of negative samples

    data = pd.read_csv('../data/course3_edge.csv')
    data = np.array(data)  # (46116,2)

    G = nx.Graph()  # 无向图
    G.add_edges_from(data)
    # print(G.number_of_nodes(), G.number_of_edges())#16714, 46116

    # node2vec = Node2Vec(G=G, len_walk=len_walk, num_walk=num_walk, workers=workers, p=p, q=q)
    # node2vec.train(embedding_size=embedding_size, window_size=window_size, iterations=iterations, batch_size=batch_size, lr=lr, negative=negative)
    test()




