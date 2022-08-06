"""
This file is used for training model based on dataset and model implementation.

TP, FP1, FP2, TN, FN are defined as below:
    TP  : True positive, center distance of ball between prediction and ground truch is smaller than 5 pixel
    FP1 : False positive, center distance of ball between prediction and ground truch is larger than 5 pixel
    FP2 : Fasle positive, if ball is not in ground truth but in prediction.
    TN  : True negative.
    FN  : False positive.

options:

  -h, --help            show this help message and exit
  --HEIGHT              height of image input(default: 288)
  --WIDTH               width of image input(default: 512)
  --epochs              number of training epochs(default: 50)
  --load_weights        path to load pre-trained weights(default: None)
  --sigma               radius of circle generated in heat map(default: 5)
  --tol                 acceptable tolerance of heat map circle center between ground truth and prediction(default: 10.0)
  --batch_size          batch size(default: 2)
  --lr                  initial learning rate(default: 1)
  --dataset DATASET     Path of dataset (merged dataset)
  --worker WORKER       Number of worker to increase speed (default: 1)
  --alpha ALPHA         Focal loss Alpha(default: 0.85)
  --gamma GAMMA         Focal loss gamma(default: 1)


Note: WE recommend you to use default opetions for first one.

"""
from utils.volleydataset import VollyDataset, VollyDatasetV2, VollyDatasetV3
from sklearn.model_selection import train_test_split
from utils.focalloss import FocalLoss, FocalLoss2
from utils.validation import outcome, evaluation
from utils.res_tracknet import ResNet_Track
from torch.utils.data import DataLoader
from logging import raiseExceptions
from torch.optim import Adadelta
import matplotlib.pyplot as plt
from utils.utils import Print
from parser import parser
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import time
import os


# Parameters
args = parser.parse_args()
dataset_path = args.dataset
BATCH_SIZE   = args.batch_size
WIDTH        = args.WIDTH
HEIGHT       = args.HEIGHT
R            = args.sigma
WORKERS      = args.worker
ALPHA        = args.alpha  
GAMMA        = args.gamma
LR           = args.lr
EPOCH        = args.epochs
TOlERANCE    = args.tol
LOAD_MODEL   = args.load_weights
START        = args.start
SAVE_PATH    = args.save_path
LOG_PATH     = args.log


print = Print(log_path=LOG_PATH)

# Check CUDA
CUDA = torch.cuda.is_available()
print("CUDA Availability: ", CUDA)
if CUDA:
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Train test split
data_train, data_val = train_test_split(pd.read_csv(dataset_path), test_size=0.03, random_state=5)

# Load Train Dataset 
volley_dataset    = VollyDatasetV3(data_train, r=R, width=WIDTH, height=HEIGHT, name='Training')
if CUDA:
    volley_dataloader = DataLoader(volley_dataset, batch_size= BATCH_SIZE, shuffle=True, num_workers=WORKERS,pin_memory= True)
else:
    volley_dataloader = DataLoader(volley_dataset, batch_size= BATCH_SIZE, shuffle=True)

# Load Validation dataset
volley_dataset_val = VollyDatasetV3(data_val, r=R, width=WIDTH, height=HEIGHT, name='Validation')


# Loading Model
model = ResNet_Track().to(device)
# model.last[6].bias.data.fill_(-3.2)

if LOAD_MODEL != 'None':
    try:
        model.load_state_dict(torch.load(LOAD_MODEL))
        model.eval()
        print("\nModel ({}) is Loaded".format(LOAD_MODEL))
    except:
        print("Problem in loading model ! ");exit()




# Loading loss function
focal_loss = FocalLoss2(alpha=ALPHA, gamma=GAMMA)

# Optimizer
optimizer   = Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# parameter for model 
loss_arr = []
accu_arr = []
best_loss = 5.0
best_accuracy = 0.0

# Training loop
torch.autograd.set_detect_anomaly(True)
for epoch in range(START,EPOCH):
    model.train()

    print(40*'='+f" {epoch+1}/{EPOCH} "+"="*40)
    epoch_time = time.time()
    
    # Iterate on off batches
    total_batches = len(volley_dataloader)
    total_loss    = 0.0
    for batch_idx, [input_image, output_label] in enumerate(volley_dataloader):
        batch_time = time.time()
        # Go to GPU
        input_image = input_image.to(device)

        # Predict
        output_pred = model(input_image)

        # Find loss 
        loss = focal_loss.forward(output_pred.cpu(), output_label.cpu())
        total_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx % 200 == 0): print("Epoch: [{}/{}]  ,Batch index: [{}/{}], Loss Value:[{:.8f}], time: [{}]".format(epoch+1, EPOCH, batch_idx+1, len(volley_dataloader),loss, time.time()-batch_time))

    loss_arr.append(total_loss/len(volley_dataloader))

    # Validation part
    model.eval()
    TP = TN = FP1 = FP2 = FN = 0
    for valid_data, valid_label in volley_dataset_val:
        valid_label = valid_label.unsqueeze(0)
        valid_data  = valid_data.unsqueeze(0)

        # Prediction of validation data
        valid_data = valid_data.to(device)
        valid_pred = model(valid_data)

        # Prepare prediction for calculate accuracy and other
        valid_pred = valid_pred.cpu().detach()
        valid_pred = valid_pred.numpy() > 0.5
        valid_pred = valid_pred.astype(np.float32)

        # Prepare label for that
        valid_label = valid_label.numpy()

        # Find TP, TN, FP12, FN
        tp, tn, fp1, fp2, fn = outcome(valid_pred, valid_label, TOlERANCE)
        TP  += tp
        TN  += tn
        FP1 += fp1
        FP2 += fp2
        FN  += fn

    # Update Learning Rate
    # if epoch > EPOCH/2:
    #     for g in optimizer.param_groups:
    #         g['lr'] = LR / 10
    #         print('\nLearning Rate Updated to ({})'.format(g["lr"]))

    
    # Find Accuracy, Recall and preciesion
    accuracy, precision, recall = evaluation(TP, TN, FP1, FP2, FN)
    accu_arr.append(accuracy)
    print("\n")
    print("Loss      : {:.5f}".format(total_loss/len(volley_dataloader)))
    print("Accuracy  : {:.5f}".format(accuracy))
    print("precision : {:.5f}".format(precision))
    print("recall    : {:.5f}".format(recall))

    print(f"TP ({TP}), TN ({TN}), FP1 ({FP1}), FP2 ({FP2}), FN ({FN})")

    # Save model
    torch.save(model.state_dict(), os.path.join(SAVE_PATH,"last_model.pt"))
    if total_loss/len(volley_dataloader) < best_loss:
        best_loss = total_loss/len(volley_dataloader)
        torch.save(model.state_dict(), os.path.join(SAVE_PATH,"best_loss_model.pt"))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(SAVE_PATH,"best_acc_model.pt"))


# Save loss array and acc array
np.savez('information.npz', loss_arr=loss_arr, accu_arr=accu_arr)

    




