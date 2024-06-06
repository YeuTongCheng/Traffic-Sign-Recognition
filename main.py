# Data Preprocessing
from google.colab import drive
drive.mount('/content/drive') # where I store the training data
# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import random
import shutil

# PyTorch dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

from google.colab import drive
drive.mount('/content/drive')

import zipfile

# file: zip file
zip_path = 'Path where I stor the data'

# creat a ZipFile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # unzip to specific
    zip_ref.extractall('/content/data')

# when unzipping finish, using ls to check unzipped files.
!ls /content/data

! unzip "path to the zip file"

import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_data_path = '/content/data/GTSRB/Final_Training/Images'
train_path = '/content/train'
test_path = '/content/test'

# Create train and test directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Splitting function
def split_data(class_name):
    class_path = os.path.join(original_data_path, class_name)
    images = [os.path.join(class_path, f) for f in os.listdir(class_path)]
    train_images, test_images = train_test_split(images, test_size=0.2)  # 20% for testing

    # Function to copy files
    def copy_files(files, destination):
        for file in files:
            shutil.copy(file, destination)

    # Create class directories in train and test
    train_class_dir = os.path.join(train_path, class_name)
    test_class_dir = os.path.join(test_path, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy files
    copy_files(train_images, train_class_dir)
    copy_files(test_images, test_class_dir)

# Iterate over each class and split data
for class_name in os.listdir(original_data_path):
    if os.path.isdir(os.path.join(original_data_path, class_name)):
        split_data(class_name)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# Data transform to convert data to a tensor and apply normalization

# augment train and validation dataset with RandomHorizontalFlip and RandomRotation
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

from PIL import Image

# Replace with a specific PPM file path
test_image_path = '/content/train/00000_00000.ppm'

try:
    img = Image.open(test_image_path)
    img.show()  # This will display the image if you are in a GUI environment
    print("PPM file opened successfully.")
except IOError as e:
    print(f"Error in opening PPM file: {e}")

# choose the training and test datasets
train_dataset = datasets.ImageFolder(root='/content/train', transform=train_transform)
testset = datasets.ImageFolder('/content/test', transform = test_transform)

# obtain training indices that will be used for validation
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
# Samples elements randomly from a given list of indices, without replacement.
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
