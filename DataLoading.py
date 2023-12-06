from torchvision.io import read_video
from torchvision.transforms import v2
import torch
from torch.utils.data import Dataset
import numpy as np
import os

def get_video_list(root_path):
    instances = []
    classes = sorted(entry.name for entry in os.scandir(root_path) if entry.is_dir())
    for target_class in classes:
        target_dir = os.path.join(root_path, target_class)
        for file in os.listdir(target_dir):
            instances.append(os.path.join(target_dir, file))
    return instances


class UCF101FullVideo(Dataset):
    # Inspired by the UCF101 pyTorch implementation. We want full videos and not divided into clips.
    def __init__(self, root, transform=None, output_format="THWC"):
        self.transform = transform
        self.root = root
        self.video_list = get_video_list(root)
        self.output_format = output_format

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        filename = self.video_list[idx]
        video, audio, info = read_video(filename, output_format=self.output_format)
        if self.transform is not None:
            video = self.transform(video)
        return video

def custom_collate(batch):
  augment = v2.Compose([
    v2.RandomResizedCrop(size=(224,224), scale=(0.5,1), antialias=True),
    v2.RandomHorizontalFlip(),
    # For ablation study
    # v2.ColorJitter()
  ])

  augmented_batch = []
  for video in batch:
      n_frames = len(video)
      idx1 = np.random.randint(low=0,high=n_frames-5, size=1)
      idx2 = np.random.randint(low=4, high=min(48, (n_frames-idx1)), size=1)
      f1f2 = torch.concatenate((video[idx1], video[idx2]))
      f1f2_augmented = augment(f1f2)
      augmented_batch.append(f1f2)
      augmented_batch.append(f1f2_augmented)

  return torch.stack(augmented_batch)[torch.randperm(len(augmented_batch))]