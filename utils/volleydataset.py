import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from .generate_heatmap import genHeatMap
from .motion_channel import motion_channel, motion_channelV2
from torch.utils.data import Dataset, DataLoader


class VollyDataset(Dataset):
    """
        This class dataset is used for dataloader function.
        csv dataset that we use is contain of this part: 
            video_path  frame_idx  cord_1_x  cord_1_y  cord_2_x  cord_2_y  cord_3_x  cord_3_y
        
        input:
            1. path to dataset csv file

    """
    # Define initial function
    def __init__(self, dataset_path, r=3, mag=1, width=512, height=288, name='training'):
        print(" ---------- Dataset is loaded ({}) ---------- ".format(name))
        if type(dataset_path) is str:
            self.dataset = pd.read_csv(dataset_path)
        else:
            self.dataset = dataset_path
        self.width   = width
        self.height  = height
        self.r       = r
        self.mag     = mag

        print("Number of frames : ", len(self.dataset))
        print("Width            : ", self.width)
        print("Height           : ", self.height)
        print("\n")


        # Define Transform list 
        # --- Define Transforms
        self.transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5,0.5,0.5],std =[0.5,0.5,0.5])
                                ])
        
        self.transform_label = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.ToTensor()
                                ])

    # getitem function
    def __getitem__(self, index):
        # Get row information
        selected_row = self.dataset.iloc[index]            

        # read video
        cap = cv2.VideoCapture(VollyDataset.resolve_letter(selected_row['video_path']))
        vid_width  = cap. get(cv2. CAP_PROP_FRAME_WIDTH )
        vid_height = cap. get(cv2. CAP_PROP_FRAME_HEIGHT)

        # First image
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_row['frame_idx'])
        _, image_1 = cap.read()

        # Secound image
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_row['frame_idx']+1)
        _, image_2 = cap.read()

        # Third image
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_row['frame_idx']+2)
        _, image_3 = cap.read()

        # Preprocess path
        image_1, image_2, image_3 = map(self.base_transform, [image_1, image_2, image_3])

        # Label arrays
        label_1 = genHeatMap(self.width, 
                             self.height, 
                             (selected_row['cord_1_x']*self.width)//vid_width,
                             (selected_row['cord_1_y']*self.height)//vid_height,self.r,self.mag
                             )
        label_2 = genHeatMap(self.width, 
                             self.height, 
                             (selected_row['cord_2_x']*self.width)//vid_width,
                             (selected_row['cord_2_y']*self.height)//vid_height,self.r,self.mag
                             )
        label_3 = genHeatMap(self.width, 
                             self.height, 
                             (selected_row['cord_3_x']*self.width)//vid_width,
                             (selected_row['cord_3_y']*self.height)//vid_height,self.r,self.mag
                             )
        
        
        # Apply transform to inputs images
        input_images        = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        input_images[:,:,0] = image_1; input_images[:,:,1] = image_2; input_images[:,:,2] = image_3
        input_images        = self.transform(input_images)

        # Apply transform to label images
        label_images        = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        label_images[:,:,0] = label_1; label_images[:,:,1] = label_2; label_images[:,:,2] = label_3
        label_images        = self.transform_label(label_images)

        return input_images, label_images


    # return len of triplets
    def __len__(self):
        return len(self.dataset)
    
    # Make gray scale image
    def base_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.width, self.height))
        return img
    
    # Change '\' to '/'
    def resolve_letter(string):
        result = ""
        for letter in string:
            if letter == '\\':
                letter = '/'
            result += letter
        
        return result


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class VollyDatasetV2(Dataset):
    """
        This class dataset is used for dataloader function.
        csv dataset that we use is contain of this part: 
            video_path  frame_idx  cord_1_x  cord_1_y  cord_2_x  cord_2_y  cord_3_x  cord_3_y
        
        input:
            1. path to dataset csv file

    """
    # Define initial function
    def __init__(self, dataset_path, r=3, mag=1, width=512, height=288, name='training'):
        print(" ---------- Dataset is loaded ({}) ---------- ".format(name))
        if type(dataset_path) is str:
            self.dataset = pd.read_csv(dataset_path)
        else:
            self.dataset = dataset_path
        self.width   = width
        self.height  = height
        self.r       = r
        self.mag     = mag

        print("Number of frames : ", len(self.dataset))
        print("Width            : ", self.width)
        print("Height           : ", self.height)
        print("\n")


        # Define Transform list 
        # --- Define Transforms
        self.transform = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
        

    # getitem function
    def __getitem__(self, index):
        # Get row information
        selected_row = self.dataset.iloc[index]            

        # read video
        cap = cv2.VideoCapture(VollyDataset.resolve_letter(selected_row['video_path']))
        vid_width  = cap. get(cv2. CAP_PROP_FRAME_WIDTH )
        vid_height = cap. get(cv2. CAP_PROP_FRAME_HEIGHT)

        # First image
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_row['frame_idx'])
        _, image_1 = cap.read()

        # Secound image
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_row['frame_idx']+1)
        _, image_2 = cap.read()

        # Third image
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_row['frame_idx']+2)
        _, image_3 = cap.read()

        # Preprocess path
        image_4 = motion_channelV2(image_1, image_2, image_3)
        image_4 = cv2.resize(image_4, (self.width, self.height))

        image_1, image_2, image_3 = map(self.base_transform, [image_1, image_2, image_3])

        # Label arrays
        label_3 = genHeatMap(self.width, 
                             self.height, 
                             (selected_row['cord_3_x']*self.width)//vid_width,
                             (selected_row['cord_3_y']*self.height)//vid_height,self.r,self.mag
                             )
        
        # Apply transform to inputs images
        input_images        = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        input_images[:,:,0] = image_1; input_images[:,:,1] = image_2; input_images[:,:,2] = image_3; input_images[:,:,3] = image_4
        input_images        = self.transform(input_images)

        # Apply transform to label images
        label_images = torch.as_tensor(np.array(label_3), dtype=torch.float64)
        label_images = label_images.unsqueeze(0)


        
        return input_images, label_images


    # return len of triplets
    def __len__(self):
        return len(self.dataset)
    
    # Make gray scale image
    def base_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.width, self.height))
        return img
    
    # Change '\' to '/'
    def resolve_letter(string):
        result = ""
        for letter in string:
            if letter == '\\':
                letter = '/'
            result += letter
        
        return result
                





######### --------------- TEST --------------- #########
# volley_dataloader = DataLoader(VollyDatasetV2('merged_dataset.csv', r=5), 
#                        batch_size= 5,
#                        shuffle=True,
#                     #    num_workers=1,
#                     #    pin_memory= True
#                        )


# for i,j in volley_dataloader:
#     img   = i[0].numpy()
#     label = j[0].numpy()
#     # print(i.shape, j.shape);exit()
#     for k in range(4):
#         X = img[k]
#         Y = label[0]
#         plt.subplot(1,2, 1)
#         plt.imshow(X)
#         plt.subplot(1,2, 2)
#         plt.imshow(Y)
#         plt.show()

# X = next(iter(volley_dataloader))
# print(X.shape)