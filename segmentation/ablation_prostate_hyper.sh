#!/bin/bash

# Declare an array of transformations to compare
dataset=prostate
# transforms=(DeepStack RandomContrast RandomGamma RandomBrightness RandomNoise \
#             RandomResolution RandomMirror RandomRotate RandomScale )
transform=MedDiffAug
augment_ratios=(3 5 10 20 30 40)
# augment_ratios=( )
# fold=1
# transforms=(DeepStack)
cuda_list=(5 5 5 )
seed=3

# # Loop through the transformations and run the config_loader.py script
for augment_ratio in ${augment_ratios[@]}
do
    echo Running with augmentation ratios: ${augment_ratio}
    echo Applying additional settings for MedDiffAug
    for fold in $( seq 0 2 )
    do
        echo 'fold  ' ${fold} '***********'
        if [[ ${fold} = 2 ]]
        then
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --is_aug \
                --fold ${fold} \
                --augment_ratio ${augment_ratio} \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models//${transform}_augment_ratio_${augment_ratio}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_augment_ratio_${augment_ratio}/fold${fold} \
                --seed ${seed} \
                --transform ${transform}
        else
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --is_aug \
                --fold ${fold} \
                --augment_ratio ${augment_ratio} \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}_augment_ratio_${augment_ratio}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_augment_ratio_${augment_ratio}/fold${fold} \
                --seed ${seed} \
                --transform ${transform} &
        fi
    done

    echo "********************************"
done

# alpha_list=(0 0.2 0.4 0.6 0.8)
# alpha_list=(0.3 0.5 )
resolution_list=(1 16 32 96 128 384 )
# Loop through the transformations and run the config_loader.py script
for resolution in ${resolution_list[@]}
do
    echo Running with hyper-resolution: ${resolution}
    echo Applying additional settings for MedDiffAug
    for fold in $( seq 0 2 )
    do
        echo 'fold  ' ${fold} '***********'
        if [[ ${fold} = 2 ]]
        then
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --is_aug \
                --fold ${fold} \
                --resolution ${resolution} \
                --augment_ratio 50 \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models//${transform}_resolution_${resolution}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_resolution_${resolution}/fold${fold} \
                --seed ${seed} \
                --transform ${transform}
        else
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --is_aug \
                --fold ${fold} \
                --augment_ratio 50 \
                --resolution ${resolution} \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}_resolution_${resolution}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_resolution_${resolution}/fold${fold} \
                --seed ${seed} \
                --transform ${transform} &
        fi
    done

    echo "********************************"
done

# alpha_list=(0 0.2 0.4 0.6 0.8)
# alpha_list=(0.3 0.5 )
alpha_list=( )
# Loop through the transformations and run the config_loader.py script
for alpha in ${alpha_list[@]}
do
    echo Running with hyper-alpha: ${alpha}
    echo Applying additional settings for MedDiffAug
    for fold in $( seq 0 2 )
    do
        echo 'fold  ' ${fold} '***********'
        if [[ ${fold} = 2 ]]
        then
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --is_aug \
                --fold ${fold} \
                --alpha ${alpha} \
                --augment_ratio 50 \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models//${transform}_alpha_${alpha}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_alpha_${alpha}/fold${fold} \
                --seed ${seed} \
                --transform ${transform}
        else
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --is_aug \
                --fold ${fold} \
                --augment_ratio 50 \
                --alpha ${alpha} \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}_alpha_${alpha}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_alpha_${alpha}/fold${fold} \
                --seed ${seed} \
                --transform ${transform} &
        fi
    done

    echo "********************************"
done


# alpha_list=(0.3 0.5 )
model_list=( )
# Loop through the transformations and run the config_loader.py script
for model in ${model_list[@]}
do
    echo Running with model: ${alpha}
    echo Applying additional settings for MedDiffAug
    for fold in $( seq 0 2 )
    do
        echo 'fold  ' ${fold} '***********'
        if [[ ${fold} = 2 ]]
        then
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --model_name ${model} \
                --is_aug \
                --fold ${fold} \
                --augment_ratio 50 \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models//${transform}_model_${model}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_model_${model}/fold${fold} \
                --seed ${seed} \
                --transform ${transform} & 
        else
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                --config ./configs/trainer/${dataset}.yaml \
                --model_name ${model} \
                --is_aug \
                --fold ${fold} \
                --augment_ratio 50 \
                --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}_model_${model}/fold${fold}.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_model_${model}/fold${fold} \
                --seed ${seed} \
                --transform ${transform} &
        fi
    done

    echo "********************************"
done

wait
python ./utils/summary.py --folder /data/datasets/DiffusionMedAug/${dataset}/seg_model/results

echo "Finished"
