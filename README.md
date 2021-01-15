# [ICLR2021 Oral]Free Lunch for Few Shot Learning: Distribution Calibration


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


## Evaluate our distribution calibration

To evaluate our distribution calibration method, run:

```eval
python evaluate_DC.py
```


