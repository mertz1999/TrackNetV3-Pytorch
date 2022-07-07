import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from generate_hearmap import genHeatMap
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
    def __init__(self, dataset_path, r=3, mag=1, width=512, height=288):
        print(" ---------- Dataset in loaded ---------- \n")
        self.dataset = pd.read_csv(dataset_path)
        self.width   = width
        self.height  = height
        self.r       = r
        self.mag     = mag

        # Define Transform list 
        # --- Define Transforms
        self.transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0,0,0],std =[1,1,1])
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
        cap = cv2.VideoCapture(selected_row['video_path'])
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





######### --------------- TEST --------------- #########
# volley_dataloader = DataLoader(VollyDataset('merged_dataset.csv'), 
#                        batch_size= 5,
#                        shuffle=True,
#                     #    num_workers=1,
#                     #    pin_memory= True
#                        )


# for i,j in volley_dataloader:
#     img   = i[0].numpy()
#     label = j[0].numpy()
#     for k in range(3):
#         X = img[k]
#         Y = label[k]
#         plt.subplot(1,2, 1)
#         plt.imshow(X+Y*255)
#         plt.subplot(1,2, 2)
#         plt.imshow(Y)
#         plt.show()
    # print(i.shape, j.shape);exit()

# X = next(iter(volley_dataloader))
# print(X.shape)