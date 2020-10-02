# Free Lunch for Few Shot Learning: Distribution Calibration


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

***Donwloading the dataset and create base/val/novel splits***:

miniImageNet
* Change directory to filelists/miniImagenet/
* Run 'source ./download_miniImagenet.sh'

CUB

* Change directory to filelists/CUB/
* Run 'source ./download_CUB.sh' 



## Training


To train the feature extractor in the paper, run this command:

```train
python train.py --dataset [miniImagenet/CUB] 
```

## Extract base/novel class features for  [miniImagenet/CUB] using pretrained extractor

- Create an empty 'checkpoints' directory.

- Run:
```save_features
python save_plk.py --dataset [miniImagenet/CUB] 
```


## Evaluate distribution calibration

To evaluate our distribution calibration method on miniImageNet/CUB, run:

```eval
python evaluate_DC.py
```


