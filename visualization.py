import re
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
from tqdm import trange
import torch.nn.functional as F
from functools import partial
import gc
import av
import joblib
import seaborn as sns
from torchvision.io import read_video
from DataLoading import *
from SiamMae import *
from LabelPropagation import *
from train import *
from utils import *

pizzaind = []
name = 'TaiChi'
#name = 'WalkingWithDog'
#name = 'YoYo'
#name = 'Swing'
#name = 'PizzaTossing'
#name = 'TrampolineJumping'
for i in range(len(train_data.video_list)):
  if train_data.video_list[i].split('/')[-2] == name:
    pizzaind.append(i)

def showimage(model_path = model_path, mask_ratio = 0.5, num_video = 100):
  model_path = model_path
  model = sim_mae_vit_tiny_patch16_dec512d8b()
  model.load_state_dict(torch.load(model_path))
  model = model.to('cuda')
  allloss = []
  inds = []
  max_value = train_data.__len__()
  num_instance = int(train_data.__len__()*0.5)
  random_sequence = np.random.choice(max_value, size=num_video, replace=False)
  NUM_INSTANCE = trange(len(random_sequence), desc='Data Index: ', leave=True)
  for index0 in NUM_INSTANCE:
    try:
      index = pizzaind[index0]
      data = train_data.__getitem__(index)[range(0,90,10)]
      data = data.to('cuda')
      losses = []
      for i in range(1,8):
        loss , pred = model.forward(data[np.newaxis,[0,i],:,:,:], mask_ratio = mask_ratio)
        pred_image = np.transpose(model.unpatchify(pred).cpu().detach().numpy(), (0, 2, 3, 1))
        losses.append(loss.cpu().detach().numpy())
      allloss.append(np.mean(losses))
      inds.append(index)
    except:
      print(f'{index}th video is shorter than 45 frames.')
      continue
    NUM_INSTANCE.set_description(f"Data Index {index0}/{num_video} - Avg. Loss: {np.mean(losses):.4f}")
  inds = np.array(inds)
  indeces = np.argsort(allloss)[:10]

  for ind in inds[np.array(indeces)]:
    data = train_data.__getitem__(ind)[range(0,90,10)]
    data = data.to('cuda')
    pred_images= []
    print(data.shape)
    for i in range(1,8):
      loss , pred = model.forward(data[np.newaxis,[0,i],:,:,:], mask_ratio = mask_ratio)
      pred_image = np.transpose(model.unpatchify(pred).cpu().detach().numpy(), (0, 2, 3, 1))
      pred_images.append(pred_image)

    x, mask, ids_restore = model.forward_encoder(data, mask_ratio = 1-mask_ratio)

    mask = torch.reshape(mask,(-1,14,14))
    resized_mask = torch.kron(mask, torch.ones((16, 16)).to('cuda'))

    data_image = np.transpose(data.cpu().detach().numpy(), (0, 2, 3, 1))

    masked_image = resized_mask[ 1:, :, :, np.newaxis].cpu() * data_image[1:]

    fig, axs = plt.subplots(3, 8, figsize=(15, 7))


    for i in range(8):
        axs[0, i].imshow(data_image[i])
        axs[0, i].axis('off')
        axs[0, 0].set_title('Raw frames', fontsize=20)

    for i in range(7):
        axs[1, i+1].imshow(masked_image[i])
        axs[1, i+1].axis('off')
        axs[1, 0].set_title('Masked frames', fontsize=20)
    axs[1, 0].axis('off')

    for i in range(7):
        axs[2, i+1].imshow(pred_images[i][0])
        axs[2, i+1].axis('off')
        axs[2, 0].set_title('Reconstructed frames', fontsize=20)
    axs[2, 0].axis('off')


    plt.tight_layout()
    plt.show()

def showseg(video_name):
  image_folder_path_0 = "/content/DAVIS/JPEGImages/480p/"
  true_annotation_path0 = '/content/DAVIS/Annotations/480p'
  i = video_name
  image_folder = os.path.join(image_folder_path_0, i)
  true_annotation_folder = os.path.join(true_annotation_path0, i)
  images = []
  merged_images = []
  merged_images2 = []
  image_files = sorted(os.listdir(image_folder))

  for image_file in image_files:
      if image_file.endswith(".jpg"):
          image_path = os.path.join(image_folder, image_file)
          images.append(cv2.imread(image_path))

  height, width, layers = images[0].shape

  for i, image in enumerate(images):
      annotation_file = image_files[i].replace(".jpg", ".png")
      annotation_path = os.path.join(image_folder, annotation_file)
      ref_path = os.path.join(true_annotation_folder, image_files[i])


      if os.path.exists(annotation_path) and os.path.exists(ref_path):
          annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
          trueannotation = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)

          
          if image.shape[:2] != annotation.shape[:2]:
              annotation = cv2.resize(annotation, (width, height))

          alpha = annotation[:, :, :] / 255.0  

          alpha = np.reshape(alpha[:, :, None],annotation.shape)  
          merged_image = cv2.addWeighted(image, 1, annotation[:, :, :3], 3, 0)

          merged_image = (merged_image * alpha + image * (1 - alpha)).astype("uint8")

          alpha2 = trueannotation / 255.0  

          alpha2 = np.reshape(alpha2[:, :, None],annotation.shape) 

          merged_image2 = cv2.addWeighted(image, 1, trueannotation[:, :, :], 3, 0)

          merged_image2 = (merged_image2 * alpha2 + image * (1 - alpha2)).astype("uint8")

          merged_images.append(merged_image)
          merged_images2.append(merged_image2)
  selected_percentages = [0, 1, 25, 50, 100]

  total_images = len(merged_images)
  selected_indices = [int((percentage / 100) * (total_images - 1)) for percentage in selected_percentages]

  
  fig, axes = plt.subplots(1, 5, figsize=(20, 4))

  
  for i, ax in enumerate(axes):
    if i < len(selected_indices):
      if i == 0:
        index = selected_indices[0]
        ax.imshow(cv2.cvtColor(merged_images2[index], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{video_name}, Reference", fontsize = 20)
        ax.set_xticks([])  
        ax.set_yticks([])  
      else:
        index = selected_indices[i]
        ax.imshow(cv2.cvtColor(merged_images[index], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{video_name}, {selected_percentages[i]}%", fontsize = 20)
        ax.set_xticks([])  
        ax.set_yticks([])  

    else:
        ax.axis('off')  

  
  plt.tight_layout()
  plt.show()