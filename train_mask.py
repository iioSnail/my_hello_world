import random

from tqdm import tqdm

from train import Train
import torch.optim as optim
import numpy as np
import torch
from util.ood_method import get_auc, ith_logit, max_logit, lof, energy, nnd
import torch.nn as nn


class TrainCL(Train):

    def __init__(self, config):
        super().__init__(config=config)

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.intent_num_criterion = torch.nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        other_params = list(set(self.mdl.parameters()) - set(self.mdl.encoder.parameters()))
        self.other_optimizer = optim.Adam(other_params, lr=self.args.lr)


    def train_epoch(self):
        self.current_epoch += 1

        all_bce_loss = []
        all_pred = []
        all_y = []

        progress = tqdm(self.train_loader, desc="Training")
        for x, y, _ in progress:
            y_one_hot = torch.zeros(len(y), self.args.ind_intent_num).to(self.args.device)
            for index, one_ys in enumerate(y):
                y_one_hot[index, one_ys] = 1
            golden_intent_num = torch.Tensor([[len(y[i])] for i in range(len(y))]).to(self.args.device)

            self.mdl.zero_grad()
            """
            logit: Shape为(batch_size, intent_num)。例如(4,3)表示batch_size为4，有3个已知意图。
                   然后进行对每个值进行sigmoid，若存在某个大于0.5，则认为其是OOD数据。
            """
            logit, _, cls_hidden_output = self.mdl(x)  # for LOF, pred_intent_num is None

            cl_loss = 0.
            # 如果是训练模式，并且对比学习的权重大于0, 则进行对比学习
            if self.args.l2 > 0:
                cl_loss = self.mdl.contrastive_learning(x, cls_hidden_output, y)
                #positive_sample = self.generate_positive_sample(y)
                #cl_loss = self.mdl.knn_contrastive_learning(cls_hidden_output, y, positive_sample)

            bce_loss = self.bce_criterion(logit, y_one_hot)  # BCEWithLogitsLoss包含sigmoid计算

            loss = bce_loss * self.args.l1 + cl_loss * self.args.l2

            loss.backward()
            self.other_optimizer.step()
            self.bert_optimizer.step()

            all_bce_loss.append(bce_loss.item())

            for one_predicted in logit.detach().cpu().numpy():
                all_pred.append((one_predicted > 0.5).nonzero()[0].tolist())
            all_y += y

            self.step += 1

            progress.set_postfix({
                "bce_loss": "%.5f" % bce_loss.item(),
                "cl_loss": "%.5f" % cl_loss.item(),
                "loss": "%.5f" % loss.item(),
            })

        bce_loss = np.mean(all_bce_loss)
        acc = sum([all_y[i] == all_pred[i] for i in range(len(all_y))]) / (len(all_y))
        print(f"[Epoch {self.current_epoch}] Train BCE Loss={bce_loss}, intent acc={acc}")

    @torch.no_grad()
    def evaluate(self, loader):
        if self.args.ood_method == 'lof':
            all_is_ood, score = lof(self.train_loader, loader, self.mdl)
        elif self.args.ood_method == 'nn_euler' or self.args.ood_method == 'nn_cosine':
            all_is_ood, score = nnd(self.train_loader, loader, self.mdl, self.args.ood_method, self.args.k)
        else:  # for energy, logit
            all_intent_num_pred = []
            all_logit = []
            all_is_ood = []

            for x, y, is_ood in loader:
                logit, pred_intent_num = self.mdl(x)
                all_logit += logit.tolist()
                all_intent_num_pred += [round(pred) for pred in pred_intent_num.squeeze(1).tolist()]
                all_is_ood += is_ood

            all_logit = np.array(all_logit)

            if self.args.ood_method == 'logit':
                if self.args.l3 != 0:
                    score = ith_logit(all_logit, all_intent_num_pred)  # Section 5.6
                else:
                    score = max_logit(all_logit)  # Main results
            elif self.args.ood_method == 'energy':
                score = energy(all_logit)

        auroc, fpr95, aupr_out, aupr_in = get_auc(all_is_ood, score)
        print(
            f'split index {self.args.split_index} auroc: {auroc}, fpr95: {fpr95}, aupr out: {aupr_out}, aupr in: {aupr_in}')

        if loader == self.valid_loader:
            # pass
            # fitlog.add_metric({"valid": {"auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}}, step=self.step, epoch=self.current_epoch)
            pass
        else:
            # fitlog.add_best_metric({"test": {"auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}})
            self.save_score(score, auroc)
            self.save_result(auroc, fpr95, aupr_out, aupr_in)

        return auroc





if __name__ == '__main__':
    exp = TrainCL('configs/cl.yaml')
    exp.train()
    exp.test()
    # fitlog.finish()
