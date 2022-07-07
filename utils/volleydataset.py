import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
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
    def __init__(self, dataset_path, width=512, height=288):
        print(" ---------- Dataset in loaded ---------- \n")
        self.dataset = pd.read_csv(dataset_path)
        self.width   = width
        self.height  = height

        # Define Transform list 
        # --- Define Transforms
        self.transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0,0,0],std =[1,1,1])
                                ])

    # getitem function
    def __getitem__(self, index):
        # Get row information
        selected_row = self.dataset.iloc[index]            

        # read video
        cap = cv2.VideoCapture(selected_row['video_path'])

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
        
        # Apply transform
        output        = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        output[:,:,0] = image_1; output[:,:,1] = image_2; output[:,:,2] = image_3
        output        = self.transform(output)
        
        return output, selected_row['video_path']


    # return len of triplets
    def __len__(self):
        return len(self.dataset)
    
    # Make gray scale image
    def base_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.width, self.height))
        return img





######### --------------- TEST --------------- #########
# volley_dataloader = DataLoader(VollyDataset('merged_dataset.csv', 1080, 720), 
#                        batch_size= 5,
#                        shuffle=True,
#                     #    num_workers=1,
#                     #    pin_memory= True
#                        )


# for i,j in volley_dataloader:
#     img = i[0].numpy()
#     for i in range(3):
#         X = img[i]
#         plt.imshow(X)
#         plt.show()
#     print(img[2].shape);exit()

# X = next(iter(volley_dataloader))
# print(X.shape)