import torch
import torch.nn as nn

from model.BERT import BERT
from torch.nn import functional as F

"""
Contrastive Learning
"""


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp  # temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = BERT(args)
        self.mlp = nn.Linear(self.args.hidden_size, args.ind_intent_num)

        self.projector = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.sim = Similarity()
        self.sim_loss = nn.CrossEntropyLoss()

    def forward(self, sens):
        pooled_output, last_hidden_output, _ = self.encoder(sens)
        logit = self.mlp(pooled_output)
        return logit, None, last_hidden_output[:, 0, :]

    def test(self, sens):
        pooled_output, _, _ = self.encoder(sens)
        return pooled_output

    def contrastive_learning(self, sens, cls_hidden_output):
        batch_size = len(sens)

        _, last_hidden_output, _ = self.encoder(sens)
        cls_hidden_output_2 = last_hidden_output[:, 0, :]

        cls_hidden_output = torch.concat([cls_hidden_output, cls_hidden_output_2], dim=0)

        # 将所有的样本再经过一次dense层
        outputs = self.projector(cls_hidden_output)

        # 计算正样本和负样本的cos相似度
        similarities = self.sim(outputs.unsqueeze(1), outputs.unsqueeze(0))

        y_true = torch.concat([torch.arange(batch_size, batch_size * 2),
                               torch.arange(0, batch_size)], dim=0).long().to(self.args.device)

        similarities = similarities - torch.eye(outputs.shape[0]) * 1e12

        # 计算对比学习的loss
        return self.sim_loss(similarities, y_true)
