# 原生LOF

超参：

```
Namespace(batch_size=4, bert_lr=1e-05, bert_path='bert-base-uncased', dataset='mixsnips', early_stop=5, epoch=100, gpu=0, hidden_size=768, l1=1, l3=0, lr=0.001, method='lof', ood_method='lof', param_dir='params/', save_threshold=0.99, seed=0, split_index=0)
```

训练过程：

```
Start training
[Epoch 1] Train BCE Loss=0.05776015936020957, intent acc=0.9349950431950149
split index 0 auroc: 0.9646713441654359, fpr95: 0.22, aupr out: 0.9758956006458335, aupr in: 0.9451801038824561
[Epoch 2] Train BCE Loss=0.004845637769265385, intent acc=0.9951848180144456
split index 0 auroc: 0.9667614475627769, fpr95: 0.215, aupr out: 0.9804161127186307, aupr in: 0.9432899988386658
[Epoch 3] Train BCE Loss=0.0025468842663372822, intent acc=0.9983005240050984
split index 0 auroc: 0.9804283604135894, fpr95: 0.105, aupr out: 0.9896325656607992, aupr in: 0.9635816454220857
[Epoch 4] Train BCE Loss=0.004543132621810545, intent acc=0.9974507860076477
split index 0 auroc: 0.9647710487444607, fpr95: 0.23, aupr out: 0.9806800333368852, aupr in: 0.9363673352368397
[Epoch 5] Train BCE Loss=0.0032645757542380536, intent acc=0.9981589010055233
split index 0 auroc: 0.9688847858197932, fpr95: 0.1825, aupr out: 0.9827820283123433, aupr in: 0.9405747171267714
[Epoch 6] Train BCE Loss=0.005410630060584489, intent acc=0.9973091630080725
split index 0 auroc: 0.9635598227474151, fpr95: 0.1875, aupr out: 0.9796145202275326, aupr in: 0.9270611576597179
[Epoch 7] Train BCE Loss=0.0007414245256481777, intent acc=0.9995751310012746
split index 0 auroc: 0.9708050221565732, fpr95: 0.1575, aupr out: 0.9841212909570758, aupr in: 0.9421557906740443
[Epoch 8] Train BCE Loss=0.005667885030923263, intent acc=0.997875655006373
split index 0 auroc: 0.9722895125553914, fpr95: 0.1575, aupr out: 0.9804176616038562, aupr in: 0.9575142467738751
Early Stop
End training
Max auroc on valid is 0.9804283604135894
```

测试结果：

```
split index 0 auroc: 0.8627523291925465, fpr95: 0.645, aupr out: 0.9188805303201765, aupr in: 0.7724831773598262
Exit without saving model parameter.
```