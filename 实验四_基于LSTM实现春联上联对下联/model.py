import random

import torch
from torch import nn

# 定义网络结构
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=0.5)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        time_step, batch_size = x.size()
        embeds = self.embeddings(x)
        output, (h_n, c_n) = self.lstm(embeds)
        output = self.linear(output.reshape(time_step * batch_size, -1))
        # 要返回所有时间点的数据，每个时间点对应一个字，也就是vocab_size维度的向量
        return output

