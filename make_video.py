import cv2 
import torch
import numpy as np
import pandas as pd 
from utils.res_tracknet import ResNet_Track
from utils.motion_channel import motion_channel, motion_channelV2, motion_channelV3
from torchvision import transforms


#################################   Functions    ####################################
# Make gray scale image
def base_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    return img

# Transform
transform = transforms.Compose([
                                    transforms.ToTensor(),
                                ])

def predict(index):
    # Third image channel
    cap.set(cv2.CAP_PROP_POS_FRAMES, index+2);_, image_3 = cap.read();image3_cp = np.copy(image_3)

    # Secound image channel
    cap.set(cv2.CAP_PROP_POS_FRAMES, index+1);_, image_2 = cap.read();image2_cp = np.copy(image_2)

    # First image channel
    cap.set(cv2.CAP_PROP_POS_FRAMES, index);_, image_1 = cap.read();image1_cp = np.copy(image_1)

    # Fourth image channel (Motion channel)
    # Preprocess path
    image_4 = motion_channelV3(image_2, image_3);image_4 = cv2.resize(image_4, (WIDTH, HEIGHT))
    image_5 = motion_channelV3(image_1, image_2);image_5 = cv2.resize(image_5, (WIDTH, HEIGHT))

    # Preprocesss images
    image_1, image_2, image_3 = map(base_transform, [image_1, image_2, image_3])

    # Apply transform to inputs images 
    input_images        = np.zeros((HEIGHT, WIDTH, 5), dtype=np.uint8)
    input_images[:,:,0] = image_1; input_images[:,:,1] = image_2; input_images[:,:,2] = image_5; input_images[:,:,3] = image_3; input_images[:,:,4] = image_4
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

    output_list = []
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
        
        output_list.append([cx_pred, cy_pred])
    
    return output_list

#####################################################################################

# Path to video file
VIDEO_PATH = "./games/8/8_06.mp4"
WIDTH      = 512
HEIGHT     = 288
MODEL_PATH = 'models/last_model (22).pt'

# Make outputs
OUTPUT     = "."+VIDEO_PATH.split(".")[-2]+"_predicted_improved.mp4"
CSV_PATH   = "."+VIDEO_PATH.split(".")[-2]+"_predicted.csv"

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


# Read video
cap = cv2.VideoCapture(VIDEO_PATH)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate  = int(cap.get(cv2.CAP_PROP_FPS))
width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

# Read CSV data
data = pd.read_csv(CSV_PATH)


# filling gaps between excatly two points
prev_frame = -1
flag = False
for i in data.Frame:
    if data['Visibility'][data.Frame == i].iloc[0] != 0 and flag == False:
        prev_frame = i
    elif data['Visibility'][data.Frame == i].iloc[0] == 0 and flag == False:
        flag = True
    elif data['Visibility'][data.Frame == i].iloc[0] == 0 and flag == True:
        flag = False 
    elif data['Visibility'][data.Frame == i].iloc[0] != 0 and flag == True:
        data['Visibility'][data.Frame == i-1] = 1 
        data['X'][data.Frame == i-1] = (data['X'][data.Frame == i].iloc[0] + data['X'][data.Frame == prev_frame].iloc[0])/2
        data['Y'][data.Frame == i-1] = (data['Y'][data.Frame == i].iloc[0] + data['Y'][data.Frame == prev_frame].iloc[0])/2
        flag = False
        print(f"Data {i-1} is Updated!")

# data.to_csv('test.csv')
# exit()

# Find lost frames another one
for i in data.Frame:
    if data['Visibility'][data.Frame == i].iloc[0] == 0 and (i > 2) and (i < (len(data) -3)):
        print(i)
        # Check for frame (i-2)(i-1)(i)
        temp_output = predict(i-2)
        if temp_output[2] != [0,0]:
            print(i, "IS UPDATED")
            data['X'][data.Frame == i] = temp_output[2][0]
            data['Y'][data.Frame == i] = temp_output[2][1]
            continue

        # Check for frame (i-1)(i)(i+1)
        temp_output = predict(i-1)
        if temp_output[1] != [0,0]:
            data['X'][data.Frame == i] = temp_output[1][0]
            data['Y'][data.Frame == i] = temp_output[1][1]

            # if data['Visibility'][data.Frame == i+1].iloc[0] == 0:
            #     data['X'][data.Frame == i+1].iloc[0] = temp_output[2][0]
            #     data['Y'][data.Frame == i+1].iloc[0] = temp_output[2][1]

            continue
        
        # Check for frame (i)(i+1)(i+2)
        temp_output = predict(i)
        if temp_output[0] != [0,0]:
            data['X'][data.Frame == i] = temp_output[0][0]
            data['Y'][data.Frame == i] = temp_output[0][1]

            # if data['Visibility'][data.Frame == i+1].iloc[0] == 0:
            #     data['X'][data.Frame == i+1].iloc[0] = temp_output[1][0]
            #     data['Y'][data.Frame == i+1].iloc[0] = temp_output[1][1]

            # if data['Visibility'][data.Frame == i+1].iloc[0] == 0:
            #     data['X'][data.Frame == i+2].iloc[0] = temp_output[2][0]
            #     data['Y'][data.Frame == i+2].iloc[0] = temp_output[2][1]

            continue



# filling gaps between excatly two points
prev_frame = -1
flag = False
for i in data.Frame:
    if data['Visibility'][data.Frame == i].iloc[0] != 0 and flag == False:
        prev_frame = i
    elif data['Visibility'][data.Frame == i].iloc[0] == 0 and flag == False:
        flag = True
    elif data['Visibility'][data.Frame == i].iloc[0] == 0 and flag == True:
        flag = False 
    elif data['Visibility'][data.Frame == i].iloc[0] != 0 and flag == True:
        data['Visibility'][data.Frame == i-1] = 1 
        data['X'][data.Frame == i-1] = (data['X'][data.Frame == i].iloc[0] + data['X'][data.Frame == prev_frame].iloc[0])/2
        data['Y'][data.Frame == i-1] = (data['Y'][data.Frame == i].iloc[0] + data['Y'][data.Frame == prev_frame].iloc[0])/2
        flag = False
        print(f"Data {i-1} is Updated!")   

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
out_vid = cv2.VideoWriter(OUTPUT, fourcc, frame_rate, (width, height),True)


frame_idx = 2
while(frame_idx <= total_frame-2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    _, image_3 = cap.read()

    cx = int(data['X'][data.Frame == frame_idx].iloc[0])
    cy = int(data['Y'][data.Frame == frame_idx].iloc[0])

    cv2.circle(image_3, (cx, cy), 5, (0,0,255), -1)

    out_vid.write(image_3)

    frame_idx += 1


out_vid.release()
cap.release()



