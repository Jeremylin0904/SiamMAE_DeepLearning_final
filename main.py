import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from urllib import request
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
from functools import partial
import gc
import av
import joblib

from timm.models.vision_transformer import PatchEmbed

from DataLoading import *
from SiamMae import *
from LabelPropagation import *
from train import *
from utils import *

# Data loading

# Change it in your branch !
root_path = '/home/sebas/UCF-101'

transforms = v2.Compose([
    v2.Resize(size=(224,224), antialias=True),
    v2.Lambda(lambd=lambda x: x/255.0)
])

train_data = UCF101FullVideo(root=root_path, output_format="TCHW",transform=transforms)
train_loader = DataLoader(train_data, 32, shuffle=True, collate_fn=custom_collate, pin_memory=True, num_workers=6)

# Model training

# Model, optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_mae_vit_tiny_path8 = sim_mae_vit_tiny_patch8_dec512d8b
model = sim_mae_vit_tiny_path8().to(device)
# Change in your branch
folder_logs = '/home/sebas/training_50_epochs_t8/logs.txt'
folder_model = '/home/sebas/training_50_epochs_t8'

num_epochs = 50
model = train(model, train_loader, folder_logs, folder_model, num_epochs=num_epochs, lr=1e-4)