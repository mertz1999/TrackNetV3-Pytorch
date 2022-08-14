
  

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
because this repository is private, you must to use ssh method for cloning. or set your github username and password for in your local git. 

Then you need to install pre-requirements. for that use this block of code:
```shell
python -m pip install -f requirements.txt
```
Also you need to install CUDA version of Pytorch from [here](https://pytorch.org/get-started/locally/) and CUDA toolkit from NVIDIA [official webpage](https://developer.nvidia.com/cuda-downloads).

For faster running is recommended to install [cuDNN](https://developer.nvidia.com/cudnn)







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