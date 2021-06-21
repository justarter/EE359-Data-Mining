import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import copy
class MyDataset(tud.Dataset):
    def __init__(self, text, word2idx, word_freqs, window_size, negative, len_walk):
        super(MyDataset, self).__init__()
        self.modified_text = torch.LongTensor([word2idx[word] for word in text])  # nn.Embedding need LongTensor
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)# 为了multinomial 转换成tensor
        self.window_size = window_size
        self.negative = negative
        self.len_walk = len_walk

    def __len__(self):
        return len(self.modified_text)

    def __getitem__(self, idx):
        center_words = self.modified_text[idx]  # center words

        walk_id = idx // self.len_walk
        walk = self.modified_text[walk_id * self.len_walk: (walk_id + 1) * self.len_walk]

        new_idx = idx % self.len_walk
        indices = [i for i in range(new_idx - self.window_size, new_idx+self.window_size+1) if i != new_idx]
        indices = [(indice + self.len_walk) % self.len_walk for indice in indices]# 不能截断后面forward会有问题，可以循环

        positive_words = walk[indices]  # positive words

        select_weight = copy.deepcopy(self.word_freqs)
        pos_real_idx = [pos_word.numpy().tolist() for pos_word in positive_words]
        select_weight[pos_real_idx] = 0
        cen_real_idx = center_words.numpy().tolist()
        select_weight[cen_real_idx] = 0
        negative_words = torch.multinomial(select_weight, self.negative * positive_words.shape[0],
                                           True)  # negative个负样本对应一个正样本

        return center_words, positive_words, negative_words
