"""
Prediction file: 
using this file for making result of pre-trained models.

Input parameters:
"""

from utils.res_tracknet import ResNet_Track
from utils.motion_channel import motion_channel, motion_channelV2
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

########################## Functions ############################
# Make gray scale image
def base_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    return img

def plt_images(img1_1, img1_2, img2_1, title):
    plt.subplot(2,2,1)
    plt.imshow(img1_1)

    plt.subplot(2,2,2)
    plt.imshow(img1_2)

    plt.subplot(2,2,3)
    plt.imshow(img2_1)

    plt.title(title)
    plt.show()


#################################################################


# Parameters
VIDEO_PATH = "./games/8/8_06.mp4"
MODEL_PATH = "./models/last_model (20).pt"
WIDTH      = 512
HEIGHT     = 288


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

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
    model.eval()
    print("\nModel ({}) is Loaded".format(MODEL_PATH))
except:
    print("Problem in loading model ! ");exit()


# Transform
transform = transforms.Compose([
                                    # transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[115.,115.,115.,0.5],std =[55.,55.,55.,10.])
                                ])

# Reading video for prodiction
cap = cv2.VideoCapture(VIDEO_PATH)
total_frame = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
frame_idx = 100
while(frame_idx < total_frame-3):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    _, image_1 = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx+1)
    _, image_2 = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx+2)
    _, image_3 = cap.read()

    # Preprocess path
    image_4 = motion_channelV2(image_1, image_2, image_3)
    image_4 = cv2.resize(image_4, (WIDTH, HEIGHT))

    image_1, image_2, image_3 = map(base_transform, [image_1, image_2, image_3])

    # Apply transform to inputs images 
    input_images        = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    input_images[:,:,0] = image_1; input_images[:,:,1] = image_2; input_images[:,:,2] = image_3; input_images[:,:,3] = image_4
    input_images        = transform(input_images).unsqueeze(0)


    # Predict this frames
    input_images = input_images.to(device)
    pred_images  = model(input_images)

    pred_images = pred_images.cpu().detach()
    # pred_images = (pred_images - torch.min(pred_images))/(torch.max(pred_images)-torch.min(pred_images))
    pred_images = pred_images.squeeze(0)#*255
    pred_images = pred_images.numpy()#.astype(np.uint8)

    # pred_images  = pred_images > 0.5
    # print(np.max(pred_images));exit()

    
    plt_images(image_3,pred_images[0],image_4, title=str(frame_idx+2))

    frame_idx += 3