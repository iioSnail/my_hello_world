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

        if self.args.cl_method == 'knn':
            self.init_cl_queue()
            self.negative_data = self.create_negative_dataset()

    def init_cl_queue(self):
        if self.args.l2 <= 0:
            return

        queue_size = self.args.queue_size

        features_queue = torch.rand(0, 768, device=self.args.device)
        labels_queue = []

        with torch.no_grad():
            print("Start initiate contrastive learning queue...")
            for x, y, _ in self.train_loader:
                _, _, cls_hidden_output = self.mdl(x)
                cls_hidden_output = torch.nn.functional.normalize(cls_hidden_output)
                features_queue = torch.cat([features_queue, cls_hidden_output])
                labels_queue.extend(y)

                if len(labels_queue) >= queue_size:
                    break

            print("Complete initiate contrastive learning queue...")

        shuffle_idx = torch.randperm(features_queue.size(0))
        features_queue = features_queue[shuffle_idx]
        labels_queue = [labels_queue[idx] for idx in shuffle_idx]

        features_queue = features_queue[: queue_size]
        labels_queue = labels_queue[: queue_size]

        self.mdl.features_queue = features_queue
        self.mdl.labels_queue = labels_queue

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
            logit, token_hidden_outputs, cls_hidden_output, mask = self.mdl(x)  # for LOF, pred_intent_num is None

            cl_loss = 0.
            # 如果是训练模式，并且对比学习的权重大于0, 则进行对比学习
            if self.args.cl_method == 'knn':
                positive_sample = self.generate_positive_sample(y)
                cl_loss = self.mdl.knn_contrastive_learning(cls_hidden_output, y, positive_sample)
            elif self.args.cl_method == 'SimCLR':
                cl_loss = self.mdl.contrastive_learning(x, cls_hidden_output)
            elif self.args.cl_method == 'label_representation':
                cl_loss = self.mdl.label_representation_contrastive_learning(token_hidden_outputs, mask, y)
            else:
                print("[WARNING] Cannot find any contrastive learning method!")

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

    def generate_positive_sample(self, labels):
        positive_num = 3
        positive_sens = []
        positive_labels = []
        for label in labels:
            candidates_samples = []
            for key in label:
                candidates_samples.extend(self.negative_data[key])
            samples = random.sample(candidates_samples,
                                    positive_num if len(candidates_samples) > positive_num else len(candidates_samples))
            for i in range(len(samples)):
                positive_labels.append(label)

            positive_sens.extend(samples)
        return {
            "sentences": positive_sens,
            "labels": positive_labels
        }

    def create_negative_dataset(self):
        negative_dataset = {}
        data = self.train_set

        for line in data:
            for x in line[1]:
                inputs = line[0]
                if x not in negative_dataset.keys():
                    negative_dataset[x] = [inputs]
                else:
                    negative_dataset[x].append(inputs)

        return negative_dataset


if __name__ == '__main__':
    exp = TrainCL('configs/cl.yaml')
    exp.train()
    exp.test()
    # fitlog.finish()
