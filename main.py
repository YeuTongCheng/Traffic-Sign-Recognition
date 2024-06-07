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

# Set up training dataset & training dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    num_workers=num_workers,shuffle=True)

# specify the image classes
classes = ['Speed limit (20km/h)',
            'Speed limit (30km/h)',
            'Speed limit (50km/h)',
            'Speed limit (60km/h)',
            'Speed limit (70km/h)',
            'Speed limit (80km/h)',
            'End of speed limit (80km/h)',
            'Speed limit (100km/h)',
            'Speed limit (120km/h)',
            'No passing',
            'No passing veh over 3.5 tons',
            'Right-of-way at intersection',
            'Priority road',
            'Yield',
            'Stop',
            'No vehicles',
            'Veh > 3.5 tons prohibited',
            'No entry',
            'General caution',
            'Dangerous curve left',
            'Dangerous curve right',
            'Double curve',
            'Bumpy road',
            'Slippery road',
            'Road narrows on the right',
            'Road work',
            'Traffic signals',
            'Pedestrians',
            'Children crossing',
            'Bicycles crossing',
            'Beware of ice/snow',
            'Wild animals crossing',
            'End speed + passing limits',
            'Turn right ahead',
            'Turn left ahead',
            'Ahead only',
            'Go straight or right',
            'Go straight or left',
            'Keep right',
            'Keep left',
            'Roundabout mandatory',
            'End of no passing',
            'End no passing veh > 3.5 tons']
classes

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    img = np.clip(img, 0, 1)  # clip values to [0, 1]
    plt.imshow(np.transpose(img, (1, 2, 0)))

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
images.shape # (number of examples: 20, number of channels: 3, pixel sizes: 32x32)

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(30, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

# Create the network
class TrafficSignal(nn.Module):
    def __init__(self):
        super(TrafficSignal, self).__init__()
        self.fc1 = nn.Linear(1024*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 43)  # Output layer with 43 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

# Create an instance of the network
model1 = TrafficSignal()
model1

# Train the network
from torch import optim

criterion = nn.NLLLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.003)

epochs = 30
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:

        images = images.view(images.shape[0], -1)  # Flatten the input image
        #Training pass
        output=model1(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(train_loader)}")


