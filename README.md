
  

# TrackNetV3 Pytoch
This repo contains modification models based on TrackNetv2 and we use Pytorch framework for development.
This repo is based on:
- [Orginal TrackNetv2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2.git)
- [Modified TrackNetv2](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2)
- [Pytorch FocalLoss](https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py)



## Requirements
- Python
- Pytorch
- CUDA
- CUDNN


## How to install
First and foremost you need to clone one of the branch that you need to train or make prediction.

``` shell
git clone --branch branch_name git@github.com:volleyanalyzer/TrackNetV3-Pytorch.git
```
because this repository is private, you must to use ssh method for cloning or set your github username and password for in your local git. 

Then you need to install pre-requirements. for that use this block of code:
```shell
python -m pip install -f requirements.txt
```
Also you need to install CUDA version of Pytorch from [here](https://pytorch.org/get-started/locally/) and CUDA toolkit from NVIDIA [official webpage](https://developer.nvidia.com/cuda-downloads).
For faster running is recommended to install [cuDNN](https://developer.nvidia.com/cudnn)

If you want to check is CUDA and GPU is working correctly in pytorch, use this block of code:
```python
import torch
CUDA = torch.cuda.is_available()
print(CUDA)
```

## Training
before start training you need to make your dataset correctly.

your dataset structure must be in this format:

```
.
└── games
    ├── 1
    │   ├── 1_01.mp4
    │   ├── 1_01_ball.csv
    │   ├── 1_02.mp4
    │   └── 1_02_ball.csv
    ├── 2
    │   └── ...
    ├── 3
    │   └── ...
    ├── 4
    │   └── ...
    └── ...

```

use our modified tracknetv2 labeling tool from here.

Also you need to merge all csv ball labeled file ***(merge_dataset.csv)***.

if you have new dataset you must yse merge_dataset.py file:
```shell
python merge_dataset.py ./games
```

for training use ***train.py*** file. This file has multiple switch and inputs that you can access them with ***-h***.
```shell
python train.py -h
```

```
usage: train.py [-h] [--HEIGHT HEIGHT] [--WIDTH WIDTH] [--start START] [--epochs EPOCHS] [--load_weights LOAD_WEIGHTS] [--save_path SAVE_PATH] [--log LOG] [--sigma SIGMA] [--tol TOL] [--batch_size BATCH_SIZE] [--lr LR] [--dataset DATASET] [--worker WORKER] [--alpha ALPHA] [--gamma GAMMA]

options:
  -h, --help            show this help message and exit
  --HEIGHT              height of image input(default: 288)
  --WIDTH               width of image input(default: 512)
  --start               Starting epoch(default: 0)
  --epochs              number of training epochs(default: 50)
  --load_weights        path to load pre-trained weights(default: None)
  --save_path           path to load pre-trained weights(default: ./models)
  --log                 path to log file(default: ./log.txt)
  --sigma               radius of circle generated in heat map(default: 5)
  --tol                 acceptable tolerance of heat map circle center between ground truth and prediction(default: 10.0)
  --batch_size          batch size(default: 16)
  --lr                  initial learning rate(default: 1)
  --dataset             Path of dataset (merged dataset)
  --worker              Number of worker to increase speed (default: 1
  --alpha               Focal loss Alpha(default: 0.85)
  --gamma               Focal loss gamma(default: 1)
```

all information will be saved based on --log option because we use this file for future to making acc, prec, ... charts.


## Inference
For predict location of ball in a video file, use ***predict_video.py***.
```
python predict_video.py VIDEO_PATH MODEL_PATH WIDTH HEIGHT
``` 
for example:
```
python predict_video.py './games/1/1_01.mp4' './models/last_model.pt' 512 288
``` 
keep in mind that this file make .csv and .mp4 file for you next to the video path. (ex. 1_01_predicted.mp4, 1_01_predicted.csv)

<br/><br/><br/>

## About Branches
We Implement multiple models to test which one is best.  <br/>
**Explain branches:** <br/>
**1. main:** <br/>
This model has **4** inputs and **1** output. one of the inputs is motion channel so we have this structure: <br/>
inputs: <br/>

> frame_1 <br/>
>  frame_2  <br/>
>  frame_3  <br/>
>  motion(frame_1, frame_2, frame_3) <br/>

outputs: <br/>
> frame_1 prediction <br/>

**2. 4in3out** <br/>
This model has **5** inputs and **3** output.  <br/>
> frame_1 <br/>
>  frame_2  <br/>
>  motion(frame_1, frame_2) <br/>
>  frame_3  <br/>
>  motion(frame_2, frame_3) <br/>

outputs: <br/>
> frame_1 prediction <br/>
> frame_2 prediction <br/>
> frame_3 prediction <br/>
> 

**3. 9in3out** <br/>
This model has **9** inputs and **3** output.  <br/>
> frame_1_RGB <br/>
>  frame_2 _RGB <br/>
>  frame_3 _RGB <br/>

outputs: <br/>
> frame_1 prediction <br/>
> frame_2 prediction <br/>
> frame_3 prediction <br/>

