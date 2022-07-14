import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--HEIGHT', type=int, default=288,
                    help='height of image input(default: 288)')
parser.add_argument('--WIDTH', type=int, default=512,
                    help='width of image input(default: 512)')
parser.add_argument('--start', type=int, default=0,
                    help='Starting epoch(default: 0)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training epochs(default: 50)')
parser.add_argument('--load_weights', type=str, default="None",
                    help='path to load pre-trained weights(default: None)')
parser.add_argument('--save_path', type=str, default="./models",
                    help='path to load pre-trained weights(default: ./models)')
parser.add_argument('--log', type=str, default="./log.txt",
                    help='path to log file(default: ./log.txt)')
parser.add_argument('--sigma', type=float, default=5,
                    help='radius of circle generated in heat map(default: 5)')
parser.add_argument('--tol', type=float, default=10.0,
                    help='''acceptable tolerance of heat map circle center between 
                            ground truth and prediction(default: 10.0)''')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size(default: 16)')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate(default: 1)')
parser.add_argument('--dataset', type=str, default='./merged_dataset.csv',
                    help='Path of dataset (merged dataset)')
parser.add_argument('--worker', type=int, default=1,
                    help='Number of worker to increase speed (default: 1')
parser.add_argument('--alpha', type=float, default=0.85,
                    help='Focal loss Alpha(default: 0.85)')
parser.add_argument('--gamma', type=float, default=1,
                    help='Focal loss gamma(default: 1)')

