#!/bin/bash

# Declare an array of transformations to compare
dataset=spleen
# transforms=(Baseline RandomBlur RandomSharp RandomAutocontrast RandomRotation \
#             RandomCrop RandomAffine AutoAugment RandAugment MedDiffAug )
# transforms=(Baseline RandomContrast RandomGamma RandomBrightness RandomNoise \
#             RandomResolution RandomMirror RandomRotate RandomScale )
# fold=1
transforms=(MedDiffAug )
cuda_list=(4 6 7)

# Loop through the transformations and run the config_loader.py script
for transform in ${transforms[@]}
do
    echo Running with transformation: ${transformation}
    for fold in $( seq 0 2 )
    do
        echo 'fold  ' ${fold} '***********'
        if [[ ${fold} = 2 ]]
        then
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/test.py \
                --config ./configs/trainer/${dataset}.yaml --fold ${fold} \
                --weight_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}/fold${fold}_best-v1.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}/fold${fold}
        else
            CUDA_VISIBLE_DEVICES=${cuda_list[${fold}]} python ./engine/test.py \
                --config ./configs/trainer/${dataset}.yaml --fold ${fold} \
                --weight_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/models/${transform}/fold${fold}_best-v1.ckpt \
                --result_dir /data/datasets/DiffusionMedAug/${dataset}/seg_model/results/${transform}/fold${fold} &
        fi
    done

    echo "********************************"
done


wait
python ./utils/summary.py --folder /data/datasets/DiffusionMedAug/${dataset}/seg_model/results

echo "Finished"
