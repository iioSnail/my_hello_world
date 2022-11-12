import torch
import torch.nn as nn

from model.BERT import BERT
from torch.nn import functional as F

"""
Contrastive Learning
"""


def euclidean_metric(a, b,temp=0.05):
    n = a.shape[0]
    a = a.expand(n, n, -1)
    b = b.expand(n, n, -1)
    logits = ((a - b)**2).sum(dim=2)
    return 1/(logits+1)/temp


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, sim_method='cos', temp=0.05):
        super().__init__()
        self.temp = temp  # temperature
        if sim_method == 'cos':
            self.cos = nn.CosineSimilarity(dim=-1)
        elif sim_method == 'euclidean':
            self.cos = euclidean_metric
        else:
            raise Exception("Please specify the similarity method.")

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = BERT(args)
        self.mlp = nn.Linear(self.args.hidden_size, args.ind_intent_num)

        self.projector = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        #self.projector = = nn.Sequential(nn.Linear(self.args.hidden_size,self.args.hidden_size), nn.ReLU(), nn.Linear(self.args.hidden_size,self.args.hidden_size))
        self.sim = Similarity(sim_method=self.args.sim_method)
        self.sim_method = self.args.sim_method
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

        cls_hidden_output = torch.cat([cls_hidden_output, cls_hidden_output_2], dim=0)

        # 将所有的样本再经过一次dense层
        outputs = self.projector(cls_hidden_output)

        # 计算每个样本与其他所有样本的相似度
        similarities = self.sim(outputs.unsqueeze(1), outputs.unsqueeze(0))

        # 因为第二次过bert直接拼到第一次的后面的，所以y_true为[128, 129, ...., 0, 1, 2, ...]
        y_true = torch.cat([torch.arange(batch_size, batch_size * 2),
                               torch.arange(0, batch_size)], dim=0).long().to(self.args.device)

        similarities = similarities - (torch.eye(similarities.shape[0]) * 1e12).to(self.args.device)

        # 计算对比学习的loss
        return self.sim_loss(similarities, y_true)
