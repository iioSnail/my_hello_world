import torch
import torch.nn as nn

from model.BERT import BERT
from torch.nn import functional as F

"""
Contrastive Learning
"""


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = BERT(args)
        self.mlp = nn.Linear(self.args.hidden_size, args.ind_intent_num)

    def forward(self, sens, training=False):
        pooled_output, _, _ = self.encoder(sens)
        logit = self.mlp(pooled_output)
        return logit, None

    def test(self, sens):
        pooled_output, _, _ = self.encoder(sens)
        return pooled_output

    def contrastive_learning(self, embeddings):
        dropout = 0.1
        if self.args.dropout:
            dropout = self.args.dropout

        # 构建正样本
        sample_0 = F.dropout(embeddings, dropout)
        sample_1 = F.dropout(embeddings, dropout)

        pass

