import torch
from torch import nn
import math
import torch.nn.functional as F


class AttentionAvg(nn.Module):

    def __init__(self, hidden_size=768, dropout=0.1):
        super(AttentionAvg, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(p=dropout) # TODO

    def forward(self, inputs, mask=None):
        query = self.W_q(inputs)
        key = self.W_k(inputs)
        value = self.W_v(inputs)

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=2)

        outputs = torch.einsum("btd,btt->bd", value, weights)
        return outputs
