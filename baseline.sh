# ood_method='logit'
# for ((split_index=0; split_index<5; ++split_index)); do
#     for ((seed=0; seed<1; ++seed)); do
#         python train_bce.py --gpu 0 --seed $seed --split_index $split_index --ood_method $ood_method --dataset multiwoz23
#         python train_bce.py --gpu 0 --seed $seed --split_index $split_index --ood_method $ood_method --dataset mixsnips
#     done
# done

# ood_method='lof'
# for ((split_index=0; split_index<5; ++split_index)); do
#     for ((seed=0; seed<5; ++seed)); do
#         python train_bce.py --gpu 0 --seed $seed --split_index $split_index --ood_method $ood_method --dataset multiwoz23
#     done
# done

ood_method='nn_euler'
#ood_method='nn_cosine'
for ((split_index=0; split_index<5; ++split_index)); do
    for ((seed=0; seed<1; ++seed)); do
        python train_bce.py --gpu 0 --method "lof" --seed $seed --split_index $split_index --ood_method $ood_method --dataset multiwoz23
        python train_bce.py --gpu 0 --method "lof" --seed $seed --split_index $split_index --ood_method $ood_method --dataset mixsnips
        python train_bce.py --gpu 0 --method "lof" --seed $seed --split_index $split_index --ood_method $ood_method --dataset atis
        python train_bce.py --gpu 0 --method "lof" --seed $seed --split_index $split_index --ood_method $ood_method --dataset FSPS
    done
done