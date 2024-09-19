#!/bin/bash


### First finetune the train data and the conduct sampling
# dataset=acl
# base_prompt="Anterior cruciate ligament, MRI"
dataset=breast
base_prompt="breast, ultrasound"

:'python ./reorganize_data.py \
       --dataset ${dataset} \
       --base_prompt "${base_prompt}" '


for i in $( seq 1 5 )
do
    echo 'fold  ' ${i} '***********'
    CUDA_VISIBLE_DEVICES=${i} python ./finetune_training.py \
                                    --prompts_path /data/datasets/DiffusionMedAug/${dataset}/images/prompt/prompts_${i}.json \
                                    --out_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}.ckpt \
                                    --epochs 2000 &
done

:' for i in $( seq 1 5 )
do
    echo 'fold  ' ${i} '***********'
    CUDA_VISIBLE_DEVICES=${i} python ./dataset_sample.py \
                                    --folder_dir /data/datasets/DiffusionMedAug/${dataset}/images/batch_0 \
                                    --model_weight_dir /data/datasets/DiffusionMedAug/${dataset}/saved_model/fold${i}.ckpt \
                                    --fold ${i} --num_samples 50 &
done'

wait

kill -9 ps -ef | grep python | awk '{print $2}'
echo "finished"