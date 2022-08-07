"""
Use this file to generate video with predicted labels.

positional arguments:
  video_path  Path to video file
  model_path  Path to model file
  Width       WIDTH
  Height      HEIGHT

"""

from utils.res_tracknet import ResNet_Track
from utils.motion_channel import motion_channel, motion_channelV2, motion_channelV3
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
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
parser.add_argument('--show_map', type=bool,
                    help='show map?', default=True)

args = parser.parse_args()


# Parameters
VIDEO_PATH = args.video_path
MODEL_PATH = args.model_path
WIDTH      = args.Width
HEIGHT     = args.Height
show_map   = args.show_map

OUTPUT     = "."+VIDEO_PATH.split(".")[-2]+"_predicted.mp4"
OUTPUT_CSV = "."+VIDEO_PATH.split(".")[-2]+"_predicted.csv"
f = open(OUTPUT_CSV, 'w')
f.write('Frame,Visibility,X,Y\n')


# Check processor
CUDA = torch.cuda.is_available()
print("CUDA Availability: ", CUDA)
if CUDA:
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Loading Model
model = ResNet_Track().to(device)
model = model.eval()
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
frame_idx = 0
start_time = time.time()
while(frame_idx <= total_frame-2):
    print(f'Reading Frame {frame_idx}')

    # Third image channel
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx+2)
        _, image_3 = cap.read()
        image3_cp = np.copy(image_3)

        # Secound image channel
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx+1)
        _, image_2 = cap.read()
        image2_cp = np.copy(image_2)

        # First image channel
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, image_1 = cap.read()
        image1_cp = np.copy(image_1)

        input_images_list = [image1_cp, image2_cp, image3_cp]

        # Fourth image channel (Motion channel)
        # Preprocess path
        image_4 = motion_channelV3(image_2, image_3)
        image_4 = cv2.resize(image_4, (WIDTH, HEIGHT))

        image_5 = motion_channelV3(image_1, image_2)
        image_5 = cv2.resize(image_5, (WIDTH, HEIGHT))

        # Preprocesss images
        image_1, image_2, image_3 = map(base_transform, [image_1, image_2, image_3])

        # Apply transform to inputs images 
        input_images        = np.zeros((HEIGHT, WIDTH, 9), dtype=np.uint8)
        input_images[:,:,0:3] = image_1; input_images[:,:,3:6] = image_2; input_images[:,:,6:9] = image_3
        input_images        = transform(input_images).unsqueeze(0)


        

        # Predict this frames
        input_images = input_images.to(device)
        pred_images  = model(input_images)


        # Postproccess
        pred_images = pred_images.cpu().detach()
        pred_images = pred_images.squeeze(0)
        pred_images = pred_images.numpy()
        out_image   = pred_images.copy()
        pred_images = pred_images > 0.5
        pred_images = pred_images * 255
        pred_images = pred_images.astype(np.uint8)

        pred_images_copy = pred_images.copy()


        for channel in range(3):
            X = pred_images_copy[channel]
            (cnts, _) = cv2.findContours(X, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            
            cv2.circle(input_images_list[channel], (cx_pred, cy_pred), 5, (0,0,255), -1)
            f.write(f'{frame_idx+channel},{0 if (cx_pred, cy_pred) == (0,0) else 1},{cx_pred},{cy_pred}\n')
            
            # plt.imshow(input_images_list[channel]);plt.show()

            out_vid.write(input_images_list[channel])
    except:
        break


    frame_idx += 3

print("\nTotal Time: ", time.time() - start_time)
out_vid.release()
cap.release()
cv2.destroyAllWindows()
