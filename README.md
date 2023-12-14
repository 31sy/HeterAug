# HeterAug

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This respository is the PyTorch implementation of the TIP 2023 paper named **"Exploring the Robustness of Human Parsers Toward Common Corruptions"** .

This paper is mainly focused on improving the model robustness under commonly corrupted conditions.

Features:

- [x] Three corruption robustness benchmarks, LIP-C, ATR-C and Pascal-Person-Part-C.
- [x] Pre-trained human parsers on three popular single person human parsing datasets.
- [x] Training and inference code.

## Requirements

```
Python >= 3.6, PyTorch >= 1.0
```

The detail environment can be referred to the HeterAug.yaml file.

## Dataset Preparation

Please download the [LIP](http://sysu-hcp.net/lip/) dataset following the below structure. 

```commandline
datasets/LIP
|--- train_images # 30462 training single person images
|--- val_images # 10000 validation single person images
|--- train_segmentations # 30462 training annotations
|--- val_segmentations # 10000 validation annotations
|--- train_id.txt # training image list
|--- val_id.txt # validation image list
```

The common corrupted images can be download from the link [LIP-C](https://pan.baidu.com/s/1VXi_YrPloA2W9S98Vu6f9w?pwd=av2n) (提取码: av2n)

```commandline
datasets/LIP-C
|--- blurs 
     |--- defocus_blur
     |--- gaussian_blur
     |--- glass_blur
     |--- motion_blur
     |--- val_segmentations
     |--- val_id.txt 
|--- digitals 
     |--- brightness
     |--- contrast
     |--- saturate
     |--- jpeg_compression
     |--- val_segmentations
     |--- val_id.txt 
|--- noises
     |--- gaussian_noise
     |--- shot_noise
     |--- impulse_noise
     |--- speckle_noise
     |--- val_segmentations
     |--- val_id.txt 
|--- weathers
     |--- snow
     |--- fog
     |--- spatter
     |--- frost
     |--- val_segmentations
     |--- val_id.txt 
|--- val_segmentations # 10000 validation annotations
|--- val_id.txt # validation image list
```


## Training

```
CUDA_VISIBLE_DEVICES=0,1 python -u train_augpolicy_mixed_noisenet_epsilon.py --batch-size 14 --gpu 0,1 \
                        --data-dir ./datasets/LIP --noisenet-prob 0.25 --log-dir 'log/LIP_heteraug' 
```

By default, the trained model will be saved in `./log/LIP_heteraug` directory. Please read the arguments for more details. 

## Evaluation on the clean data

```
python evaluate.py --model-restore [CHECKPOINT_PATH] --data-dir ./datasets/LIP
```

CHECKPOINT_PATH should be the path of trained model. If you want to testing with flipping, you should add --flip.

## Evaluation on the corrupted data

```
python evaluate_c.py --model-restore [CHECKPOINT_PATH] --data-dir ./datasets/LIP-C/blurs/ --severity-level 5 --corruption_type 'glass_blur' 2>&1 | tee ./'SCHP_glass_blur.log'
```

The pre-trained models can be downloaded from the link [pre-trained models](链接：https://pan.baidu.com/s/1fS5LAYG99VKoAy9Y4Cy4rw?pwd=im5i ) (提取码：im5i )

## The robustness benchmark construction.

You can use the code [imagecorruption](https://github.com/bethgelab/imagecorruptions) to generate the corrupted validation images.

There are 16 different types of image corruptions. Each image corruption employs 5 severity levels. All corruption types can be categorized into four groups, *i.e.*, blur (defocus, gaussian, motion, glass), noise (gaussian, impulse, shot, speckle), digital (brightness, contrast, saturate, JPEG compression), and weather (fog, frost, snow, spatter).

## Citation

Please cite our work if you find this repo useful in your research.

```latex
@article{zhang2023heter,
  author={Zhang, Sanyi and Cao, Xiaochun and Wang, Rui and Qi, Guo-Jun and Zhou, Jie},
  journal={IEEE Transactions on Image Processing}, 
  title={Exploring the Robustness of Human Parsers Toward Common Corruptions}, 
  year={2023},
  volume={32},
  number={},
  pages={5394-5407},
  doi={10.1109/TIP.2023.3313493}}
```



Our code is implemented on the Self correction human parsing model, please refer to the code link： [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing).
