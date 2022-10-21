# 最近邻（欧拉）

超参：

```
Namespace(batch_size=4, bert_lr=1e-05, bert_path='bert-base-uncased', dataset='mixsnips', early_stop=5, epoch=100, gpu=0, hidden_size=768, l1=1, l3=0, lr=0.001, method='lof', ood_method='nn_euler', param_dir='params/', save_threshold=0.99, seed=0, split_index=0)
```

训练过程：

```
Start training
[Epoch 1] Train BCE Loss=0.05109422604084312, intent acc=0.9443421611669736
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 81.02it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 78.38it/s]
split index 0 auroc: 0.9627141802067947, fpr95: 0.1825, aupr out: 0.9787212763150887, aupr in: 0.9253428455729519
[Epoch 2] Train BCE Loss=0.00404387595088905, intent acc=0.9973091630080725
Build Training Data Representations: 100%|██████████| 1766/1766 [00:22<00:00, 78.29it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 75.39it/s]
split index 0 auroc: 0.9584047267355983, fpr95: 0.21, aupr out: 0.9746587163329077, aupr in: 0.9257025344524773
[Epoch 3] Train BCE Loss=0.005977661300402838, intent acc=0.9954680640135958
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 80.55it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 75.96it/s]
split index 0 auroc: 0.9757200886262924, fpr95: 0.0875, aupr out: 0.9838823877426099, aupr in: 0.9566394465900713
[Epoch 4] Train BCE Loss=0.00396752551670671, intent acc=0.9977340320067979
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 82.03it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 78.35it/s]
split index 0 auroc: 0.9699593796159527, fpr95: 0.1175, aupr out: 0.9798780973528036, aupr in: 0.9476238248849833
[Epoch 5] Train BCE Loss=0.0034956012136593212, intent acc=0.9975924090072228
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 82.65it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 78.93it/s]
split index 0 auroc: 0.9642060561299852, fpr95: 0.1675, aupr out: 0.97944119189559, aupr in: 0.9290054455213157
[Epoch 6] Train BCE Loss=2.2445787839996315e-05, intent acc=1.0
Build Training Data Representations: 100%|██████████| 1766/1766 [00:20<00:00, 85.19it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 81.52it/s]
split index 0 auroc: 0.9678064992614477, fpr95: 0.1175, aupr out: 0.9794894071616351, aupr in: 0.93644952434507
[Epoch 7] Train BCE Loss=0.005705903876517377, intent acc=0.9977340320067979
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 82.76it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:04<00:00, 60.59it/s]
split index 0 auroc: 0.9779246676514033, fpr95: 0.0975, aupr out: 0.9773767799149635, aupr in: 0.9692413365265874
[Epoch 8] Train BCE Loss=0.0022475019652920024, intent acc=0.9990086390029741
Build Training Data Representations: 100%|██████████| 1766/1766 [00:20<00:00, 84.48it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 80.88it/s]
split index 0 auroc: 0.9926957163958642, fpr95: 0.025, aupr out: 0.9953819885092292, aupr in: 0.9895648999145685
[Epoch 9] Train BCE Loss=5.6313968145410496e-06, intent acc=1.0
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 83.97it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 79.97it/s]
split index 0 auroc: 0.9921418020679468, fpr95: 0.025, aupr out: 0.9948928026146551, aupr in: 0.9890191433803168
[Epoch 10] Train BCE Loss=5.177265887700406e-06, intent acc=1.0
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 81.30it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 80.22it/s]
split index 0 auroc: 0.994165435745938, fpr95: 0.025, aupr out: 0.9961889928978757, aupr in: 0.9922789679086781
[Epoch 11] Train BCE Loss=0.004950660215222353, intent acc=0.9980172780059482
Build Training Data Representations: 100%|██████████| 1766/1766 [00:21<00:00, 83.99it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 81.02it/s]
split index 0 auroc: 0.9890915805022156, fpr95: 0.0225, aupr out: 0.9919030097211807, aupr in: 0.98457973832025
[Epoch 12] Train BCE Loss=0.0025583786782819914, intent acc=0.9990086390029741
Build Training Data Representations: 100%|██████████| 1766/1766 [00:20<00:00, 84.94it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 79.96it/s]
split index 0 auroc: 0.9898633677991137, fpr95: 0.03, aupr out: 0.9944603475368412, aupr in: 0.9804095161883982
[Epoch 13] Train BCE Loss=0.002588495137203058, intent acc=0.998867016003399
Build Training Data Representations: 100%|██████████| 1766/1766 [00:20<00:00, 84.73it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 82.29it/s]
split index 0 auroc: 0.9896418020679469, fpr95: 0.0325, aupr out: 0.9918358777164838, aupr in: 0.986403034526532
[Epoch 14] Train BCE Loss=0.0022802563994022513, intent acc=0.9987253930038238
Build Training Data Representations: 100%|██████████| 1766/1766 [00:20<00:00, 86.43it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 82.26it/s]
split index 0 auroc: 0.9901624815361891, fpr95: 0.0325, aupr out: 0.9919325417619251, aupr in: 0.987460797603077
[Epoch 15] Train BCE Loss=1.1426903651513236e-05, intent acc=1.0
Build Training Data Representations: 100%|██████████| 1766/1766 [00:20<00:00, 85.93it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 82.16it/s]
split index 0 auroc: 0.9912444608567208, fpr95: 0.02, aupr out: 0.9929914792029538, aupr in: 0.9886100316974809
Early Stop
End training
Max auroc on valid is 0.994165435745938
```

验证结果：

```
Build Training Data Representations: 100%|██████████| 1766/1766 [00:20<00:00, 86.07it/s]
Build Test Data Representations: 100%|██████████| 261/261 [00:03<00:00, 85.46it/s]
split index 0 auroc: 0.9239052795031055, fpr95: 0.3875, aupr out: 0.9521039557332043, aupr in: 0.889982263187264
Exit without saving model parameter.
```


# 最近邻（余弦）

超参：

```
Namespace(batch_size=4, bert_lr=1e-05, bert_path='bert-base-uncased', dataset='mixsnips', early_stop=5, epoch=100, gpu=0, hidden_size=768, l1=1, l3=0, lr=0.001, method='lof', ood_method='nn_cosine', param_dir='params/', save_threshold=0.99, seed=0, split_index=0)
```

训练过程：

```
Start training
[Epoch 1] Train BCE Loss=0.05776015936020957, intent acc=0.9349950431950149
Build Training Data Representations: 100%|██████████| 1766/1766 [00:26<00:00, 67.42it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:04<00:00, 65.03it/s]
split index 0 auroc: 0.958040989660266, fpr95: 0.21, aupr out: 0.9679141212826555, aupr in: 0.9233629179065542
[Epoch 2] Train BCE Loss=0.0061375360959779815, intent acc=0.9957513100127461
Build Training Data Representations: 100%|██████████| 1766/1766 [00:25<00:00, 68.42it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:04<00:00, 64.91it/s]
split index 0 auroc: 0.9508142540620383, fpr95: 0.2325, aupr out: 0.9712785809395541, aupr in: 0.8924358365168901
[Epoch 3] Train BCE Loss=0.0007356502683612954, intent acc=0.9994335080016995
Build Training Data Representations: 100%|██████████| 1766/1766 [00:25<00:00, 69.24it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:04<00:00, 65.58it/s]
split index 0 auroc: 0.9115288035450517, fpr95: 0.37, aupr out: 0.943128223885086, aupr in: 0.8632146048497068
[Epoch 4] Train BCE Loss=0.0040968877798241285, intent acc=0.9968842940093471
Build Training Data Representations: 100%|██████████| 1766/1766 [00:25<00:00, 69.25it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.04it/s]
split index 0 auroc: 0.9619220827178729, fpr95: 0.16, aupr out: 0.9730661494438115, aupr in: 0.9362415007847733
[Epoch 5] Train BCE Loss=0.0016782321169617172, intent acc=0.9995751310012746
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 70.99it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.63it/s]
split index 0 auroc: 0.9738312407680946, fpr95: 0.1425, aupr out: 0.9841911953476183, aupr in: 0.956611387858557
[Epoch 6] Train BCE Loss=0.002695669304731844, intent acc=0.9987253930038238
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 71.47it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 67.83it/s]
split index 0 auroc: 0.9710579763663221, fpr95: 0.17, aupr out: 0.9845347866609571, aupr in: 0.9458272467442921
[Epoch 7] Train BCE Loss=0.006711747443559429, intent acc=0.996742671009772
Build Training Data Representations: 100%|██████████| 1766/1766 [00:26<00:00, 67.06it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.70it/s]
split index 0 auroc: 0.9595439438700148, fpr95: 0.2125, aupr out: 0.970434886958272, aupr in: 0.9367221059132118
[Epoch 8] Train BCE Loss=0.002906808369326096, intent acc=0.9991502620025492
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 70.90it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 67.61it/s]
split index 0 auroc: 0.9725886262924668, fpr95: 0.15, aupr out: 0.9823503318910334, aupr in: 0.9483929432733661
[Epoch 9] Train BCE Loss=0.003038755933004276, intent acc=0.9983005240050984
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 70.81it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 70.31it/s]
split index 0 auroc: 0.9765509601181683, fpr95: 0.1125, aupr out: 0.9831759100212683, aupr in: 0.9655137737462439
[Epoch 10] Train BCE Loss=0.0021069746727477973, intent acc=0.9987253930038238
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 72.01it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 69.29it/s]
split index 0 auroc: 0.9613238552437224, fpr95: 0.225, aupr out: 0.9773675357526659, aupr in: 0.9370900680224126
[Epoch 11] Train BCE Loss=0.0034415140776947253, intent acc=0.9991502620025492
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 71.36it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.21it/s]
split index 0 auroc: 0.966573116691285, fpr95: 0.1975, aupr out: 0.980517883281416, aupr in: 0.9490536034843253
[Epoch 12] Train BCE Loss=0.0011551280142200779, intent acc=0.9991502620025492
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 71.11it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:04<00:00, 67.23it/s]
split index 0 auroc: 0.9727769571639586, fpr95: 0.115, aupr out: 0.9832866694740976, aupr in: 0.9542752871334941
[Epoch 13] Train BCE Loss=0.0025251325141956026, intent acc=0.9990086390029741
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 71.57it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.74it/s]
split index 0 auroc: 0.9830225258493352, fpr95: 0.0925, aupr out: 0.9904991608080224, aupr in: 0.9711518619031618
[Epoch 14] Train BCE Loss=0.0010143174739619526, intent acc=0.9994335080016995
Build Training Data Representations: 100%|██████████| 1766/1766 [00:25<00:00, 70.29it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.01it/s]
split index 0 auroc: 0.9865214180206794, fpr95: 0.065, aupr out: 0.9924781132006548, aupr in: 0.9768485901205273
[Epoch 15] Train BCE Loss=0.0013636280930741968, intent acc=0.9995751310012746
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 72.90it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 70.13it/s]
split index 0 auroc: 0.966246307237814, fpr95: 0.1775, aupr out: 0.9806094753167816, aupr in: 0.9457360732173558
[Epoch 16] Train BCE Loss=0.0021691103974930937, intent acc=0.9994335080016995
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 71.88it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 69.17it/s]
split index 0 auroc: 0.9778581979320532, fpr95: 0.125, aupr out: 0.9865694916445685, aupr in: 0.9660960578550715
[Epoch 17] Train BCE Loss=0.0005644567307185673, intent acc=0.9995751310012746
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 71.01it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.97it/s]
split index 0 auroc: 0.9686133677991138, fpr95: 0.1825, aupr out: 0.98193826582725, aupr in: 0.9414860645112122
[Epoch 18] Train BCE Loss=0.003113429967270995, intent acc=0.9991502620025492
Build Training Data Representations: 100%|██████████| 1766/1766 [00:25<00:00, 68.03it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:04<00:00, 67.42it/s]
split index 0 auroc: 0.9778028064992614, fpr95: 0.12, aupr out: 0.9869656596855124, aupr in: 0.9652465135015865
[Epoch 19] Train BCE Loss=0.0021913561706207933, intent acc=0.9992918850021243
Build Training Data Representations: 100%|██████████| 1766/1766 [00:25<00:00, 70.40it/s]
Build Test Data Representations: 100%|██████████| 270/270 [00:03<00:00, 68.99it/s]
split index 0 auroc: 0.9801015509601182, fpr95: 0.1125, aupr out: 0.9879215029741354, aupr in: 0.9705929322331999
Early Stop
End training
Max auroc on valid is 0.9865214180206794
```

测试结果：

```
Build Training Data Representations: 100%|██████████| 1766/1766 [00:24<00:00, 71.92it/s]
Build Test Data Representations: 100%|██████████| 261/261 [00:03<00:00, 69.21it/s]
split index 0 auroc: 0.9029871894409938, fpr95: 0.44, aupr out: 0.9391522998188808, aupr in: 0.8442806376164711
Exit without saving model parameter.
```