#!/bin/bash

# Declare an array of transformations to compare
dataset=prostate
# transforms=(DeepStack RandomContrast RandomGamma RandomBrightness RandomNoise \
#             RandomResolution RandomMirror RandomRotate RandomScale )
transforms=(Baseline MedDiffAug)
# augment_ratios=(3 5 10 20 30 40)
augment_ratios=( )
# fold=1
# transforms=(DeepStack)
cuda_list=(6 6 6 )
seed=3

# alpha_list=(0.3 0.5 )
# model_list=("2DResUNet" "ResNet50UNet" "2DAttentionUnet"  "2DUNETR" "2DSwinUNETR")
model_list=("2DUNet" )

for model in ${model_list[@]}
do  
    echo Running with model: ${alpha}
    echo Applying additional settings for MedDiffAug
    for transform in ${transforms[@]}
    do  
        if [[ ${transform} = MedDiffAug || ${transform} = MedDiffAug_Plus ]]
        then
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
                        --transform ${transform}
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
        else
            for fold in $( seq 0 2 )
            do
                echo 'fold  ' ${fold} '***********'
                if [[ ${fold} = 2 ]]
                then
                    CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                        --config ./configs/trainer/${dataset}.yaml \
                        --model_name ${model} \
                        --fold ${fold} \
                        --augment_ratio 50 \
                        --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models//${transform}_model_${model}/fold${fold}.ckpt \
                        --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_model_${model}/fold${fold} \
                        --seed ${seed} \
                        --transform ${transform}
                else
                    CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                        --config ./configs/trainer/${dataset}.yaml \
                        --model_name ${model} \
                        --fold ${fold} \
                        --augment_ratio 50 \
                        --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}_model_${model}/fold${fold}.ckpt \
                        --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}_model_${model}/fold${fold} \
                        --seed ${seed} \
                        --transform ${transform} &
                fi
            done
        fi
    done
    echo "********************************"
done

wait
python ./utils/summary.py --folder /data/datasets/DiffusionMedAug/${dataset}/seg_model/results

echo "Finished"
