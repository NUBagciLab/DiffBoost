## DiffusionMedAug
# DiffBoost: Enhancing Medical Image Segmentation via Text-Guided Diffusion Model
Large-scale, big-variant, high-quality data are crucial for developing robust and successful deep-learning models for medical applications since they potentially enable better generalization performance and avoid overfitting. However, the scarcity of high-quality labeled data always presents significant challenges. This paper proposes a novel approach to address this challenge by developing controllable diffusion models for medical image synthesis, called DiffBoost. We leverage recent diffusion probabilistic models to generate realistic and diverse synthetic medical image data that preserve the essential characteristics of the original medical images by incorporating edge information of objects to guide the synthesis process. In our approach, we ensure that the synthesized samples adhere to medically relevant constraints and preserve the underlying structure of imaging data. Due to the random sampling process by the diffusion model, we can generate an arbitrary number of synthetic images with diverse appearances. To validate the effectiveness of our proposed method, we conduct an extensive set of medical image segmentation experiments on multiple datasets, including Ultrasound breast (+13.87\%), CT spleen (+0.38\%), and MRI prostate (+7.78\%), achieving significant improvements over the baseline segmentation methods. The promising results demonstrate the effectiveness of our DiffBoost for medical image segmentation tasks and show the feasibility of introducing a first-ever text-guided diffusion model for general medical image segmentation tasks. With carefully designed ablation experiments, we investigate the influence of various data augmentations, hyper-parameter settings, patch size for generating random merging mask settings, and combined influence with different network architectures.

## Guidance
This project is designed for introducing diffusion model to augment medical dataset. This can be applied for classification or segmentation task.

### Setup
Please create the environments as follow:
```
conda env create -f environment.yaml
conda activate medgconrtol
```

Datasets can be download from the original data source according to the license. All preprocess file can be found under [preprocess](./segmentation/preprocess/preprocessor.py). 

### Step 1: Train text-diffusion model for medical image
We release our RadImageNet pretrain weights <del> [here](https://drive.google.com/drive/folders/1dFitnVITUlDovC8XgLl1h12xGk13T3i8?usp=sharing).</del>

As shown in the paper, we need to finetune the model on downstream tasks. Take prostate dataset as example
```
cd ./diffusion/ControlNet
bash ./seg_finetune_sampling_prostate.sh
```
### Step 2: Apply for downstream medical tasks

```
cd ./segmentation
bash ./compare_single_prostate.sh
```
### Pretrained Weights
The pretrained weight <del> is available at [here]() </del> due to the data requirements of the RadImageNet team.
### citation

please consider cite our work if you find it is helpful
```
@article{zhang2023emit,
  title={Emit-diff: Enhancing medical image segmentation via text-guided diffusion model},
  author={Zhang, Zheyuan and Yao, Lanhong and Wang, Bin and Jha, Debesh and Keles, Elif and Medetalibeyoglu, Alpay and Bagci, Ulas},
  journal={arXiv preprint arXiv:2310.12868},
  year={2023}
}

```
