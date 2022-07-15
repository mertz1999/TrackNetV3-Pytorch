"""
Use this file to generate video with predicted labels.

positional arguments:
  video_path  Path to video file
  model_path  Path to model file
  Width       WIDTH
  Height      HEIGHT

"""

from utils.res_tracknet import ResNet_Track
from utils.motion_channel import motion_channel, motion_channelV2
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import cv2


########################## Functions ############################
# Make gray scale image
def base_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    return img

##################################################################

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('video_path', type=str,
                    help='Path to video file')
parser.add_argument('model_path', type=str,
                    help='Path to model file')
parser.add_argument('Width', type=int,
                    help='WIDTH')
parser.add_argument('Height', type=int,
                    help='HEIGHT')

args = parser.parse_args()


# Parameters
VIDEO_PATH = args.video_path
MODEL_PATH = args.model_path
WIDTH      = args.Width
HEIGHT     = args.Height

OUTPUT     = "."+VIDEO_PATH.split(".")[-2]+"_predicted.mp4"


# Check processor
CUDA = torch.cuda.is_available()
print("CUDA Availability: ", CUDA)
if CUDA:
    # torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Loading Model
model = ResNet_Track().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
    model.eval()
    print("\nModel ({}) is Loaded".format(MODEL_PATH))
except:
    print("Problem in loading model ! ");exit()


# Transform
transform = transforms.Compose([
                                    transforms.ToTensor(),
                                ])


# Reading video for prodiction
cap = cv2.VideoCapture(VIDEO_PATH)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate  = int(cap.get(cv2.CAP_PROP_FPS))
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

# Output Video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
out_vid = cv2.VideoWriter(OUTPUT, fourcc, frame_rate, (width, height),True)


# Read frame by frame
frame_idx = 2
start_time = time.time()
while(frame_idx <= total_frame-2):
    print(f'Reading Frame {frame_idx}')

    # Third image channel
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    _, image_3 = cap.read()
    image3_cp = np.copy(image_3)

    # Secound image channel
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-1)
    _, image_2 = cap.read()

    # First image channel
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-2)
    _, image_1 = cap.read()

    # Fourth image channel (Motion channel)
    image_4 = motion_channelV2(image_1, image_2, image_3)
    image_4 = cv2.resize(image_4, (WIDTH, HEIGHT))

    # Preprocesss images
    image_1, image_2, image_3 = map(base_transform, [image_1, image_2, image_3])

    # Apply transform to inputs images 
    input_images        = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    input_images[:,:,0] = image_1; input_images[:,:,1] = image_2; input_images[:,:,2] = image_3; input_images[:,:,3] = image_4
    input_images        = transform(input_images).unsqueeze(0)


    

    # Predict this frames
    input_images = input_images.to(device)
    pred_images  = model(input_images)


    # Postproccess
    pred_images = pred_images.cpu().detach()
    pred_images = pred_images.squeeze(0)
    pred_images = pred_images.numpy()
    pred_images = pred_images > 0.5
    pred_images = pred_images[0] * 255
    pred_images = pred_images.astype(np.uint8)


    #h_pred
    (cnts, _) = cv2.findContours(pred_images.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    max_area_idx = 0
    try:
        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
        for i in range(len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx]

        (cx_pred, cy_pred) = (
            int((target[0] + target[2] / 2) * (width/WIDTH)),
            int((target[1] + target[3] / 2) * (height/HEIGHT)),
            )
    except:
        (cx_pred, cy_pred) = (0,0)
    
    

    cv2.circle(image3_cp, (cx_pred, cy_pred), 5, (0,0,255), -1)

    out_vid.write(image3_cp)

    frame_idx += 1

print("\nTotal Time: ", time.time() - start_time)
out_vid.release()
cap.release()
cv2.destroyAllWindows()
