import copy

import numpy as np
import torch
import torch.nn as nn

from model.BERT import BERT
from torch.nn import functional as F

"""
Contrastive Learning
"""


def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x


def euclidean_metric(a, b, temp=0.05):
    n = a.shape[0]
    a = a.expand(n, n, -1)
    b = b.expand(n, n, -1)
    logits = ((a - b) ** 2).sum(dim=2)
    return 1 / (logits + 1) / temp


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
        self.encoder_k = copy.deepcopy(self.encoder)  # Momentum Encoder
        self.mlp = nn.Linear(self.args.hidden_size, args.ind_intent_num)

        self.projector = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.projector_k = copy.deepcopy(self.projector)  # Momentum Encoder Projector

        self.sim = Similarity(sim_method=self.args.sim_method)
        self.sim_method = self.args.sim_method
        self.sim_loss = nn.CrossEntropyLoss()

        self.cl_loss_fct = nn.CrossEntropyLoss()

    def forward(self, sens):
        pooled_output, last_hidden_output, _ = self.encoder(sens)
        logit = self.mlp(pooled_output)
        return logit, None, last_hidden_output[:, 0, :]

    def test(self, sens):
        pooled_output, _, _ = self.encoder(sens)
        return pooled_output

    def contrastive_learning(self, sens, cls_hidden_output, labels):
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

    def knn_contrastive_learning(self, cls_hidden_output, labels, positive_sample):

        with torch.no_grad():
            self.update_encoder_k()

            pooled_output, _, _ = self.encoder_k(positive_sample['sentences'])
            update_keys = self.projector_k(pooled_output)
            update_keys = l2norm(update_keys)
            self._dequeue_and_enqueue(update_keys, positive_sample['labels'])

        liner_q = self.projector(cls_hidden_output)
        liner_q = l2norm(liner_q)

        logits_con = self.select_pos_neg_sample(liner_q, labels)

        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long, device=self.args.device) # 构造labels，与moco相同，正样本都是在0位置
            loss_con = self.cl_loss_fct(logits_con, labels_con)
            return loss_con
        else:
            return 0.

        print()
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

    def update_encoder_k(self):
        m = self.args.m
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def _dequeue_and_enqueue(self, keys, label):
        batch_size = keys.shape[0]

        self.labels_queue = label + self.labels_queue[batch_size:]
        self.features_queue = torch.cat([keys, self.features_queue[batch_size:]])

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.labels_queue  # K。 队列中样本的label
        feature_queue = self.features_queue.clone().detach()  # K * hidden_size。队列中样本的特征向量

        # 1. expand label_queue and feature_queue to batch_size * K
        batch_size = len(label_q)
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1])  # batch_size * K * hidden_size

        # 2.caluate sim. (16, 768)*(16,6500,768)->(16, 6500)。求每个样本和队列中其他样本的相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3. get index of postive and neigative
        pos_mask_index = []
        for batch_label in label_q:
            item_mask_index = []
            for queue_label in label_queue:
                item_mask_index.append(batch_label == queue_label)
            pos_mask_index.append(item_mask_index)
        pos_mask_index = torch.tensor(pos_mask_index).to(self.args.device)  # 找出队列中哪些是正样本
        neg_mask_index = ~ pos_mask_index  # 找出队列中的负样本

        # 4.another option
        # feature_value = cos_sim.masked_select(pos_mask_index)
        # pos_sample = torch.full_like(cos_sim, -np.inf)
        # pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)  # 从cos_sim挑出负样本
        neg_sample = torch.full_like(cos_sim, -np.inf)
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)  # 将负样本按照neg_mask_index的位置进行防止。
        # 在上面三行代码完成后，neg_sample的shape依然为16, 6500，但是在正样本的位置上，全是-inf。

        # 5.topk
        pos_mask_index = pos_mask_index.int()
        """
        pos_number的shape为(16)，表示这16个样本在队列中正样本的数量。
        例如：[293, 349, 362, 309, 302, 293, 277, 310, 362, 362, 372, 362, 320, 277, 307, 372]
        表示第一个样本在队列中存在293个正样本
        """
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = cos_sim.topk(pos_min, dim=-1)  # 以正样本最少的为准，找出前n个正样本。加这句的目的是防止有些正样本可能不足topk个(25个)
        # pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.args.cl_k]  # self.topk = 25 # 找出每个样本的前25个正样本（相似度最高的前25个）
        # pos_sample_top_k = pos_sample[:, 0:self.top_k]
        # pos_sample_last = pos_sample[:, -1]
        ##pos_sample_last = pos_sample_last.view([-1, 1])
        pos_sample = pos_sample_top_k
        # pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.contiguous().view([-1, 1])  # 到这里，就挑出了16*25个正样本。

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_topk = min(pos_min, self.args.cl_k)
        neg_sample = neg_sample.repeat([1, neg_topk])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.args.T  # 除以温度参数
        return logits_con
