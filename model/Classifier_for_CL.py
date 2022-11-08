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
        return self.cos(x, y)  # / self.temp


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = BERT(args)
        self.mlp = nn.Linear(self.args.hidden_size, args.ind_intent_num)

        self.projector = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.sim = Similarity()
        self.sim_loss = nn.BCELoss()

    def forward(self, sens):
        pooled_output, _, _ = self.encoder(sens)
        logit = self.mlp(pooled_output)
        return logit, None, pooled_output

    def test(self, sens):
        pooled_output, _, _ = self.encoder(sens)
        return pooled_output

    def contrastive_learning(self, sens, pooled_output):
        dropout = 0.1  # 对比学习的dropout概率
        if self.args.dropout:
            dropout = self.args.dropout

        batch_size = len(sens)

        # 获取句子的input_ids
        input_ids, attention_mask = self.encoder.get_inputs(sens)
        input_ids, attention_mask = input_ids.to(self.args.device), attention_mask.to(self.args.device)

        # 构建正样本，对其进行两次dropout
        sample_1, sample_2 = input_ids.clone(), input_ids.clone()
        sample_1[torch.rand(input_ids.size()).to(self.args.device) < dropout] = 0
        sample_2[torch.rand(input_ids.size()).to(self.args.device) < dropout] = 0

        # 使用bert对dropout后的正样本进行编码
        sample_1, _ = self.encoder.forward2(sample_1, attention_mask)
        sample_2, _ = self.encoder.forward2(sample_2, attention_mask)

        # 构建负样本。
        sample_3 = pooled_output.clone()
        # 将sample3随机打乱，但是同一个位置不能再次出现
        sample_4 = pooled_output.clone()[(torch.tensor(range(batch_size)) +
                                          torch.randint(1, batch_size, (batch_size,))) % batch_size]

        # 将所有的样本再经过一次dense层
        z = self.projector(torch.concat([sample_1, sample_2, sample_3, sample_4], dim=0))

        # 计算正样本和负样本的cos相似度
        positive_sim = self.sim(z[:batch_size], z[batch_size:batch_size * 2])
        negative_sim = self.sim(z[batch_size * 2: batch_size * 3], z[batch_size * 3:])

        # 计算对比学习的loss
        return self.sim_loss(F.relu(torch.concat([positive_sim, negative_sim])),
                             torch.concat([torch.full((4,), 1), torch.full((4,), -1)]).float())
