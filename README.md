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
python train.py --dataset [miniImagenet/CUB] 
```

## Extract and save features

- Create an empty 'checkpoints' directory.

- Run:
```save_features
python save_plk.py --dataset [miniImagenet/CUB] 
```
## Or you can directly download the extracted features/pretrained models from the link:
https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing

***If you chose to download the extracted features, you are not required to download the dataset and train the network. If then, all you have to do to reproduce the results is to download the features and run evaluate_DC.py. (So easy, right? ^ - ^)***

After downloading the extracted features, please adjust your file path according to the code.


## Evaluate our distribution calibration

To evaluate our distribution calibration method, run:

```eval
python evaluate_DC.py
```

If our paper is useful for your research, please cite our paper:

```
@inproceedings{
  yang2021free,
  title={Free Lunch for Few-shot Learning:  Distribution Calibration},
  author={Shuo Yang and Lu Liu and Min Xu},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=JWOiYxMG92s}
}
```

