# [ICLR2021 Oral] Free Lunch for Few-Shot Learning: Distribution Calibration

paper link: https://openreview.net/forum?id=JWOiYxMG92s

zhihu link: https://zhuanlan.zhihu.com/p/344531704

![](illustration.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

***Donwload the dataset and create base/val/novel splits***:

miniImageNet
* Change directory to filelists/miniImagenet/
* Run 'source ./download_miniImagenet.sh'

CUB

* Change directory to filelists/CUB/
* Run 'source ./download_CUB.sh' 



## Train feature extractor


To train the feature extractor in the paper, run this command:

```train
python train.py --dataset [miniImagenet/CUB] --train_aug
```

## Extract and save features

- Create an empty 'checkpoints' directory.

- Run:
```save_features
python save_plk.py --dataset [miniImagenet/CUB] 
```
## Or you can directly download the extracted features/pretrained models from the link:
https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing

***Actually, all our algorithm is built upon the extracted features (We perform data augmentation in the feature space). The training procedure and the pretrained backbone are both irrelevant to our method. The pretrained model and the extracted features we used are the same as the reference work 'S2M2' (Their project page [https://github.com/nupurkmr9/S2M2_fewshot](url)). You can reproduce our work by just simply applying evaluate_DC.py on the provided features. Or you can apply our method on your own model.***

After downloading the extracted features, please adjust your file path according to the code.


## Evaluate our distribution calibration

To evaluate our distribution calibration method, run:

```eval
python evaluate_DC.py
```

## Citation

If our paper is useful for your research, please cite our paper:

```
@inproceedings{
yang2021free,
title={Free Lunch for Few-shot Learning:  Distribution Calibration},
author={Yang, Shuo and Liu, Lu and Xu, Min},
booktitle={International Conference on Learning Representations (ICLR)},
year={2021},
}
```

## Reference

[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)

[https://github.com/nupurkmr9/S2M2_fewshot](url)

