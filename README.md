
  

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

outputs: <br/>
> frame_1 prediction <br/>

**2. 4in3out** <br/>
This model has **5** inputs and **3** output.  <br/>
> frame_1 <br/>
>  frame_2  <br/>
>  motion(frame_1, frame_2) <br/>
>  frame_3  <br/>
>  motion(frame_2, frame_3) <br/>
<br/>
outputs: <br/>
> frame_1 prediction <br/>
> frame_2 prediction <br/>
> frame_3 prediction <br/>
> 
<br/>
**3. 9in3out** <br/>
This model has **9** inputs and **3** output.  <br/>
> frame_1_RGB <br/>
>  frame_2 _RGB <br/>
>  frame_3 _RGB <br/>
<br/>
outputs: <br/>
> frame_1 prediction <br/>
> frame_2 prediction <br/>
> frame_3 prediction <br/>


## Requirements
- Python
- Pytorch
- CUDA
- CUDNN

## How to install