#!/bin/bash

# Declare an array of transformations to compare
dataset=breast
# transforms=(DeepStack RandomContrast RandomGamma RandomBrightness RandomNoise \
#             RandomResolution RandomMirror RandomRotate RandomScale )
# transforms=(Baseline MedDiffAug DeepStack RandomContrast RandomGamma RandomBrightness RandomNoise \
#              RandomResolution RandomMirror RandomRotate RandomScale )
# fold=1
# transforms=(Baseline MedDiffAug )
# transforms=(MedDiffAug_Plus )
transforms=(RandomMirror )
cuda_list=( 5 5 5 )
seed=3

transform=MedDiffAug
folds=(0 1 2 )
seed=3


# Loop through the transformations and run the config_loader.py script
for transform in ${transforms[@]}
do
    echo Running with transformation: ${transformation}
    if [[ ${transform} = MedDiffAug || ${transform} = MedDiffAug_Plus ]]
    then
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
                    --augment_ratio 50 \
                    --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models//${transform}/fold${fold}.ckpt \
                    --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}/fold${fold} \
                    --seed ${seed} \
                    --transform ${transform}
            else
                CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                    --config ./configs/trainer/${dataset}.yaml \
                    --is_aug \
                    --fold ${fold} \
                    --augment_ratio 50 \
                    --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}/fold${fold}.ckpt \
                    --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}/fold${fold} \
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
                    --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}/fold${fold}.ckpt \
                    --result /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}/fold${fold} \
                    --fold ${fold} \
                    --seed ${seed} \
                    --transform ${transform}
            else
                CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/merge_train.py \
                    --config ./configs/trainer/${dataset}.yaml \
                    --model_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}/fold${fold}.ckpt \
                    --result /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}/fold${fold} \
                    --fold ${fold} \
                    --seed ${seed} \
                    --transform ${transform} &
            fi
        done
    fi

    echo "********************************"
done


wait
python ./utils/summary.py --folder /data/datasets/DiffusionMedAug/${dataset}/seg_model/results

echo "Finished"
