"""
This file is used for training model based on dataset and model implementation.

"""

from utils.volleydataset import VollyDataset
from utils.res_tracknet import ResNet_Track
from utils.focalloss import FocalLoss, FocalLoss2
from torch.utils.data import DataLoader
from torch.optim import Adadelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
import time

# Parameters
dataset_path = './merged_dataset.csv'
BATCH_SIZE   = 1
WIDTH        = 512
HEIGHT       = 288
R            = 3
WORKERS      = 1
ALPHA        = 0.85
GAMMA        = 2
LR           = 0.01
EPOCH        = 50

# Train test split
data_train, data_val = train_test_split(pd.read_csv(dataset_path), test_size=0.05, random_state=0)

# Load Dataset 
volley_dataset    = VollyDataset(data_train, r=R, width=WIDTH, height=HEIGHT)
volley_dataloader = DataLoader(volley_dataset, 
                       batch_size= BATCH_SIZE,
                       shuffle=True,
                    #    num_workers=WORKERS,
                    #    pin_memory= True
                       )

# Check CUDA
CUDA = torch.cuda.is_available()
print("CUDA Availability: ", CUDA)
if CUDA:
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# Loading Model
model = ResNet_Track().to(device)
model.last[6].bias.data.fill_(-3.2)


# Loading loss function
focal_loss = FocalLoss2(alpha=ALPHA, gamma=GAMMA)

# Optimizer
optimizer   = Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# parameter for model 
loss_arr = []
accu_arr = []

# Training loop
model.train()
torch.autograd.set_detect_anomaly(True)
for epoch in range(EPOCH):
    print(80*'=')
    before_time = time.time()
    
    # Iterate on off batches
    total_batches = len(volley_dataloader)
    total_loss    = 0.0
    for batch_idx, [input_image, output_label] in enumerate(volley_dataloader):
        # Go to GPU
        input_image = input_image.to(device)

        # Predict
        output_pred = model(input_image)

        # Find loss 
        loss = focal_loss.forward(output_pred.cpu(), output_label.cpu())
        total_loss += loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # print(loss.grad);exit()
        optimizer.step()

        print(output_pred.shape)

