import torch
import torch.nn as nn

from model.BERT import BERT
from model.attn import AttentionAvg


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = BERT(args)
        self.softmax = nn.Softmax(dim=1)
        # 输入特征数为bert的hidden_size, 输出特征数为1，即输出一个数字，表示该语料包含的意图数。
        self.mlp_intent_num = nn.Linear(self.args.hidden_size, 1)
        self.mlp_intent_num_2 = nn.Linear(self.args.hidden_size * 2, 1)
        self.attention_avg = AttentionAvg()

        """
        输入特征数为bert的hidden_size
        输出特征数为已知意图数量（in-distribution intent num）
        """
        self.query = nn.Linear(self.args.hidden_size, self.args.ind_intent_num, bias=False)
        if self.args.method == 'bce':
            self.mlp_logit = nn.Parameter(torch.rand(self.args.ind_intent_num, self.args.hidden_size, requires_grad=True).unsqueeze(0))

    @staticmethod
    def masked_softmax(x, m=None, axis=-1):
        if len(m.size()) == 2:
            m = m.unsqueeze(1)
        if m is not None:
            m = m.float()
            x = x * m
        e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
        if m is not None:
            e_x = e_x * m
        softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
        return softmax

    def forward(self, sens):
        pooled_output, output, mask = self.encoder(sens)

        if self.args.method == 'aik_plus':
            attention_output = self.attention_avg(output)
            intent_num = self.mlp_intent_num_2(torch.concat([pooled_output, attention_output], dim=1))
        else:
            intent_num = self.mlp_intent_num(pooled_output)
        """
        output: bert输出的h_0, h_1, ..., h_n。 Shape为(batch_size, token_num, hidden_size)，
                例如Shape为(4, 16, 768)表示4个语料，每个预料16个token，每个token的向量表示为768维
        weight: 每个token对应每个意图的权重。Shape为(batch_size, token_num, intent_num)。
                例如Shape为(4, 16, 3)表示4个预料，每个预料16个token，共三个意图。
        """
        weight = self.query(output)  # weight: (batch_size, token_num, intent_num)
        weight = torch.transpose(weight, 1, 2)  # weight: (batch_size, intent_num, token_num)
        """
        进行softmax对分布进行归一化。
        例如，对于[<cls>, 今天, 天气, 怎么样, ?, <pad>, <pad>]这个sequence的结果可能为：
             "查天气"意图: [0.01(<cls>), 0.09(今天), 0.8(天气), 0.09(怎么样), 0.01(?), 0(<pad>), 0(<pad>)]
             "导航" 意图: [0.2(<cls>), 0.2(今天), 0.2(天气), 0.2(怎么样), 0.2(?), 0(<pad>), 0(<pad>)]
             "打电话"意图: [0.2(<cls>), 0.2(今天), 0.2(天气), 0.2(怎么样), 0.2(?), 0(<pad>), 0(<pad>)]
        对于"查天气"意图，`天气`token的权重很大。但对于“导航”和“打电话”意图所有的token权重都差不多，因为这个语料本来也不包含这俩意图。                 
        """
        weight = self.masked_softmax(weight, mask)  # weight: (batch_size, intent_num, token_num)
        # 根据token进行加权平均。
        rep = torch.bmm(weight, output)  # representation: (batch, intent, hidden)

        if self.args.method == 'bce':
            logit = torch.sum(self.mlp_logit * rep, dim=2)
            return logit, intent_num
        else:
            return rep, intent_num
