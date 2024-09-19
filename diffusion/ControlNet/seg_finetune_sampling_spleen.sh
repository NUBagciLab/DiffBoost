#!/bin/bash


### First finetune the train data and the conduct sampling
# dataset=acl
# base_prompt="Anterior cruciate ligament, MRI"
# dataset=prostate
# base_prompt="prostate, MRI"
dataset=spleen
base_prompt="Spleen, CT"
cuda_list=(0 0 0)
epoch=400


for i in $( seq 0 2 )
do
echo 'fold  ' ${i} '***********' using cuda ${cuda_list[${i}]}
if [[ ${i} = 2 ]]
then
    CUDA_VISIBLE_DEVICES=${cuda_list[${i}]} python ./finetune_segmentation_training.py \
                                --dataset_dir /data/datasets/DiffusionMedAug/${dataset}/processed/2DSliceEdge \
                                --out_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}/model.ckpt \
                                --fold ${i} \
                                --epochs ${epoch}
else
    CUDA_VISIBLE_DEVICES=${cuda_list[${i}]} python ./finetune_segmentation_training.py \
                                --dataset_dir /data/datasets/DiffusionMedAug/${dataset}/processed/2DSliceEdge \
                                --out_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}/model.ckpt \
                                --fold ${i} \
                                --epochs ${epoch} &
fi
done

wait 

for i in $( seq 0 2 )
do
    echo 'sample fold  ' ${i} '***********' using cuda ${cuda_list[${i}]}
    if [[ ${i} = 2 ]]
    then
        CUDA_VISIBLE_DEVICES=${cuda_list[${i}]} python ./segmentation_sample.py \
                                    --folder_dir /data/datasets/DiffusionMedAug/${dataset}/processed/2DSliceEdge \
                                    --model_weight_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}/model.ckpt \
                                    --fold ${i} --num_samples 50 
    else
        CUDA_VISIBLE_DEVICES=${cuda_list[${i}]} python ./segmentation_sample.py \
                                    --folder_dir /data/datasets/DiffusionMedAug/${dataset}/processed/2DSliceEdge \
                                    --model_weight_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}/model.ckpt \
                                    --fold ${i} --num_samples 50 &
        sleep 10
    fi
done

wait

# kill -9 ps -ef | grep python | awk '{print $2}'
echo "finished"