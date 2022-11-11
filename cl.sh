ood_method_list=("nn_euler" "nn_cosine")
l2_list=(0.2 0)  # 对比学习损失的权重
bert_dropout_list=(0.1)
sim_method_list=("cos" "euclidean")
dataset_list=("multiwoz23" "mixsnips" "atis" "FSPS")

for ood_method in "${ood_method_list[@]}"; do
    for sim_method in "${sim_method_list[@]}"; do
        for dataset in "${dataset_list[@]}"; do
            for l2 in "${l2_list[@]}"; do
                for bert_dropout in "${bert_dropout_list[@]}"; do
                    for ((split_index=0; split_index<5; ++split_index)); do
                        for ((seed=0; seed<5; ++seed)); do
                            python train_cl.py --gpu 0 --method cl --seed $seed --split_index $split_index --ood_method $ood_method --dataset $dataset --l2 $l2 --sim_method $sim_method --bert_dropout $bert_dropout
                        done  # seed
                    done # split_index
                done # bert_dropout
            done # l2_list
        done  # dataset
    done  # sim_method
done  # ood_method


# lof
for ood_method in "${ood_method_list[@]}"; do
    for dataset in "${dataset_list[@]}"; do
        for ((split_index=0; split_index<5; ++split_index)); do
            for ((seed=0; seed<5; ++seed)); do
                    python train_bce.py --gpu 0 --method lof --seed $seed --split_index $split_index --ood_method $ood_method --dataset $dataset
            done  # seed
        done # split_index
    done  # dataset
done  # ood_method
