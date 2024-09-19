#!/bin/bash


### First finetune the train data and the conduct sampling
# dataset=acl
# base_prompt="Anterior cruciate ligament, MRI"
# dataset=prostate
# base_prompt="prostate, MRI"
dataset=spleen
base_prompt="Spleen, CT"
cuda_list=(4 6 7)

i=2
echo 'fold  ' ${i} '***********'
CUDA_VISIBLE_DEVICES=${cuda_list[${i}]} python ./segmentation_sample.py \
                                --folder_dir /data/datasets/DiffusionMedAug/${dataset}/processed/2DSliceEdge \
                                --model_weight_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}/model.ckpt \
                                --fold ${i} --num_samples 50 &

i=1
echo 'fold  ' ${i} '***********'
CUDA_VISIBLE_DEVICES=${cuda_list[${i}]} python ./segmentation_sample.py \
                                --folder_dir /data/datasets/DiffusionMedAug/${dataset}/processed/2DSliceEdge \
                                --model_weight_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}/model.ckpt \
                                --fold ${i} --num_samples 50 


# kill -9 ps -ef | grep python | awk '{print $2}'
echo "finished"