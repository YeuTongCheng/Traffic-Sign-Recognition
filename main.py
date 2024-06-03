# **Team 43: Final Project - German Traffic Sign Recognition Benchmark (GTSRB)**
***Business problem***: Electric vehicles have undergone significant advancements, with a notable
emphasis on intelligence as a key driver for their development. One prominent trend contributing
to this progress is the integration of autonomous driving systems, which are increasingly
becoming core components of electric vehicles. Recognizing that environmental perception
serves as the foundation for intelligent planning and secure decision-making in autonomous
vehicles, our focus is on advancing the innovation of traffic signs recognition. This enhancement
aims to elevate the reliability and safety of automated vehicles by bolstering their ability to
perceive and respond to traffic signs.

***Data Source***: The German Traffic Sign Recognition Benchmark (GTSRB) --- The official training and
test dataset contains images and annotations, three sets of different HOG features and Haar-like
features, Hue histograms

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
