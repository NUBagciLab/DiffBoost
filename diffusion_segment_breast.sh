cd ./diffusion/ControlNet
bash ./seg_finetune_sampling_breast.sh
# bash ./once_bash.sh
cd ../segmentation
bash ./compare_single_breast.sh
