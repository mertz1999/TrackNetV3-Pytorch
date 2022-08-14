
  

# TrackNetV3 Pytoch
This repo contains modification models based on TrackNetv2 and we use Pytorch framework for development.

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

outputs:
> frame_1 prediction

**2. 4in3out**
This model has **5** inputs and **3** output. 
> frame_1
>  frame_2 
>  motion(frame_1, frame_2)
>  frame_3 
>  motion(frame_2, frame_3)

outputs:
> frame_1 prediction
> frame_2 prediction
> frame_3 prediction
> 

**3. 9in3out**
This model has **9** inputs and **3** output. 
> frame_1_RGB
>  frame_2 _RGB
>  frame_3 _RGB

outputs:
> frame_1 prediction
> frame_2 prediction
> frame_3 prediction


## Requirements
- Python
- Pytorch
- CUDA
- CUDNN

## How to install