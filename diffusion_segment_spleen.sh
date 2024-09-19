cd ./diffusion/ControlNet
bash ./seg_finetune_sampling_spleen.sh
# bash ./once_bash.sh
cd ../segmentation
bash ./compare_single_spleen.sh