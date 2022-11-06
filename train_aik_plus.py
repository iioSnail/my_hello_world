import copy

from train import Train
# import fitlog
import torch
from util.ood_method import get_auc, mahalanobis
from util.training_util import init_mmc_center
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class TrainAIKPlus(Train):

    def __init__(self, config):
        super().__init__(config=config)

        if not self.args.use_mmc:
            self.mu = torch.rand(self.args.ind_intent_num, self.args.hidden_size).to(self.args.device)
            self.mu.requires_grad = True
        else:  # Section 5.5
            assert self.args.dataset == 'multiwoz23'
            self.mu = init_mmc_center(self.args, self.ind_set)

        self.intent_num_criterion = torch.nn.MSELoss()
        other_params = list(set(self.mdl.parameters()) - set(self.mdl.encoder.parameters()))
        if not self.args.use_mmc:
            other_params += [self.mu]
        self.other_optimizer = optim.Adam(other_params, lr=self.args.lr)

    def pos_loss(self, output, y):
        """
        :param output: 网络输出的每个语料对于每种意图的向量表示
        :param y: 每个语料实际的意图。
                  例如[[1], [0,1]]，表示2个语料。第一个语料包含1个意图，意图为1
                  第二个语料包含两个意图，意图分别为0,1
        """
        golden_center = torch.tensor([]).to(self.args.device)
        golden_output = torch.tensor([]).to(self.args.device)
        for index, one_y in enumerate(y):
            # 获取y中的每个意图的意图中心点
            golden_center = torch.cat((golden_center, self.mu[one_y]), dim=0)
            # 获取网络输出对应真实意图的向量表示
            golden_output = torch.cat((golden_output, output[index][one_y]), dim=0)
        # batch_size: 并非dataloader的batch_size，而是这批语料中包含的意图总数
        bsz = len(golden_center)
        # 让网络输出的意图向量表示尽可能的接近真实意图中心
        return torch.pow(golden_output - golden_center, 2).sum() / (2 * bsz)
        
    def neg_loss(self, output, y):
        """
        :param output: 网络输出的每个语料对于每种意图的向量表示
        :param y: 每个语料实际的意图。
                  例如[[1], [0,1]]，表示2个语料。第一个语料包含1个意图，意图为1
                  第二个语料包含两个意图，意图分别为0,1
        """
        wrong_output = torch.tensor([]).to(self.args.device)
        wrong_center = torch.tensor([]).to(self.args.device)
        for index, one_y in enumerate(y):
            # 真实意图之外的意图
            wrong_y = [i for i in range(self.args.ind_intent_num) if i not in one_y]
            # 获取非真实意图的意图表示
            wrong_output = torch.cat((wrong_output, output[index][wrong_y]), dim=0)
            # 获取非真实意图的意图中心点
            wrong_center = torch.cat((wrong_center, self.mu[wrong_y]), dim=0)
        # batch_size: 为 batch_size*已知意图数 - 这批语料中包含的意图总数
        bsz = len(wrong_output)
        # margin为300，relu相当于max(0, x)
        return F.relu(self.args.margin - torch.pow(wrong_output - wrong_center, 2).sum(dim=1)).sum() / (2 * bsz)

    def train_epoch(self):
        self.current_epoch += 1
        
        all_pos_loss = []
        all_neg_loss = []
        all_intent_num_y = []
        all_intent_num_pred = []

        for x, y, _ in self.train_loader:
            self.mdl.zero_grad()
            output, pred_intent_num = self.mdl(x)
            golden_intent_num = torch.Tensor([[len(y[i])] for i in range(len(y))]).to(self.args.device)
            
            pos_loss = self.pos_loss(output, y)
            neg_loss = self.neg_loss(output, y)
            intent_num_loss = self.intent_num_criterion(pred_intent_num, golden_intent_num)
            loss = pos_loss * self.args.l1 + neg_loss * self.args.l2 + intent_num_loss * self.args.l3

            loss.backward()
            self.bert_optimizer.step()
            self.other_optimizer.step()

            if not self.args.use_mmc:
                self.mu.grad.zero_()
            
            all_pos_loss.append(pos_loss.item())
            all_neg_loss.append(neg_loss.item())
            self.step += 1

            all_intent_num_y += [len(y[i]) for i in range(len(y))]
            all_intent_num_pred += [round(pred) for pred in pred_intent_num.squeeze(1).tolist()]

        intent_num_acc = sum([all_intent_num_y[i] == all_intent_num_pred[i] for i in range(len(all_intent_num_pred))])  / (len(all_intent_num_pred))
        pos_loss = np.mean(all_pos_loss)
        neg_loss = np.mean(all_neg_loss)
        print(f"[Epoch {self.current_epoch}] pos loss: {pos_loss} neg loss: {neg_loss} intent num acc: {intent_num_acc}")

    @torch.no_grad()
    def evaluate(self, loader):
        score, all_is_ood, intent_num_acc = mahalanobis(self.train_loader, loader, self.mdl)
        auroc, fpr95, aupr_out, aupr_in = get_auc(all_is_ood, score)
        
        print(f'split index {self.args.split_index}, intent_num_acc: {intent_num_acc}, auroc: {auroc}, fpr95: {fpr95}, aupr out: {aupr_out}, aupr in: {aupr_in}')

        if loader == self.valid_loader:
            # fitlog.add_metric({"valid": {"intent_num_acc": intent_num_acc, "auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}}, step=self.step, epoch=self.current_epoch)
            pass
        else:
            # fitlog.add_best_metric({"test": {"intent_num_acc": intent_num_acc, "auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}})
            self.save_score(score, auroc)
            self.save_result(auroc, fpr95, aupr_out, aupr_in, intent_num_acc)

        return intent_num_acc, auroc

    def train(self):
        max_auroc = 0
        max_auroc_epoch = -1
        max_intent_num_accuracy = 0
        print("Start training")
        while self.current_epoch < self.args.epoch:
            self.mdl.train()
            self.train_epoch()

            self.mdl.eval()
            intent_num_acc, auroc = self.evaluate(self.valid_loader)

            if auroc > max_auroc:
                max_auroc = auroc
                max_intent_num_accuracy = intent_num_acc
                max_auroc_epoch = self.current_epoch
                self.model_parameter = copy.deepcopy(self.mdl.state_dict())
            # elif max_auroc != 0 and (self.current_epoch - max_auroc_epoch == self.args.early_stop):
            #     print("Early Stop")
            #     break
            elif max_intent_num_accuracy != 0 and (self.current_epoch - max_auroc_epoch == self.args.early_stop):
                print("Early Stop")
                break

        print("End training")
        print(f"Max auroc on valid is {max_auroc}")
    
if __name__ == '__main__':
    exp = TrainAIK('configs/aik.yaml')
    exp.train()
    exp.test()
    # fitlog.finish()