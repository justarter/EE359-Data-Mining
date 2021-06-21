import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

class MyWord2Vec(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super(MyWord2Vec, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.v_embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.u_embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

    def forward(self, center_words, positive_words, negative_words):
        center_embedding = self.v_embedding(center_words)  # [batch_size, embed_size]
        center_embedding = center_embedding.unsqueeze(axis=2)  # [batch_size, embed_size, 1]

        positive_embedding = self.u_embedding(positive_words)  # [batch_size, (window * 2), embed_size]
        negative_embedding = self.u_embedding(negative_words)  # [batch_size, (window * 2 * K), embed_size]

        pos_dot = torch.bmm(positive_embedding, center_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(axis=2)  # [batch_size, (window * 2)]
        pos_loss = F.logsigmoid(pos_dot).sum(axis=1)  # [batch_size]

        neg_dot = torch.bmm(negative_embedding, center_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(axis=2)  # [batch_size, (window * 2 * K)]
        neg_loss = F.logsigmoid(-neg_dot).sum(axis=1)  # [batch_size]

        loss = -1*(pos_loss + neg_loss)
        return loss

    def store_embedding(self):
        _embeddings = self.v_embedding.weight.cpu().detach().numpy()
        return _embeddings
