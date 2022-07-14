"""
This file is used for making chart form log file and save them
in results folder.


positional arguments:
  log_path    Path to log file

options:
  -h, --help  show this help message and exit
"""


import matplotlib.pyplot as plt
import argparse


# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('log_path', type=str,
                    help='Path to log file')
args = parser.parse_args()


# Read log file
log_file = open(args.log_path, 'r')
Lines = log_file.readlines()


loss_arr = []
accu_arr = []
prec_arr = []
reca_arr = []

loss_in_batches = []
for line in Lines:
    line_list = line.split(' ')
    if line_list[0] == 'Loss':
        loss_arr.append(float((line_list[-1])[:-1]))
    elif line_list[0] == 'Accuracy':
        accu_arr.append(float((line_list[-1])[:-1]))
    elif line_list[0] == 'precision':
        prec_arr.append(float((line_list[-1])[:-1]))
    elif line_list[0] == 'recall':
        reca_arr.append(float((line_list[-1])[:-1]))

    if line_list[0] =='Epoch:':
        loss_in_batches.append(float(line_list[7][7:-2]))


results = {
    "Loss"      : loss_arr, 
    "Accuracy"  : accu_arr, 
    "Precision" : prec_arr, 
    "Recall"    : reca_arr, 
    "Batch_loss": loss_in_batches
    }


for result in results:
    fig, ax = plt.subplots()
    ax.plot(range(len(results[result])), results[result], 'b')

    ax.set(xlabel='Epoch' if result!="Batch_loss" else 'Batches', ylabel='Values', title=f'{result} Values' )

    if result=="Batch_loss" or result=="Loss" :ax.set_yscale('log')


    fig.savefig(f"./results/{result}.png")

    print(f"{result} fiqure is saved!")
    # plt.show()