import os
import matplotlib.pyplot as plt
from LabelPropagation import  read_list_frames, read_list_labels
from tqdm import tqdm
import numpy as np 
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score

def show_segm(videos_path, labels_path, video_name, frame_num, model_name):
  frame_path = os.path.join(videos_path, video_name, frame_num) + '.jpg'
  segm_pred_path = os.path.join(videos_path, model_name, video_name, frame_num) + '.png'
  segm_gt_path = os.path.join(labels_path, video_name, frame_num) + '.png'

  frame = plt.imread(frame_path)
  segm_pred = plt.imread(segm_pred_path)
  segm_gt = plt.imread(segm_gt_path)

  fig, axs = plt.subplots(1, 2, figsize=(10,10))
  axs[0].imshow(frame, cmap='gray')
  axs[0].imshow(segm_pred, cmap='gray', alpha=0.5)
  axs[0].title.set_text('predicted segmentation')

  axs[1].imshow(frame, cmap='gray')
  axs[1].imshow(segm_gt, cmap='gray', alpha=0.5)
  axs[1].title.set_text('ground truth segmentation')

  fig.show()

def score_one_vid(videos_path, labels_path, video_name, model_name):
  list_frames = read_list_frames(os.path.join(videos_path, video_name))
  list_segs = read_list_frames(os.path.join(videos_path, model_name, video_name))
  list_labels = read_list_labels(os.path.join(labels_path, video_name))
  n = len(list_segs)
  scores = np.zeros((n-1,2))
  for i in tqdm(range(1, n)):
    pred_seg = Image.open(list_segs[i])
    real_seg = Image.open(list_labels[i])
    pred_seg = np.asarray(pred_seg).reshape(-1)
    real_seg = np.asarray(real_seg).reshape(-1)
    scores[i-1,0] = jaccard_score(real_seg, pred_seg, average='macro')
    scores[i-1,1] = f1_score(real_seg, pred_seg, average='macro')

  return np.mean(scores, axis=0)

def eval_model(list_videos, videos_path, labels_path):
  scores = []
  for video in tqdm(list_videos):
    list_frames = read_list_frames(os.path.join(videos_path, video))
    list_segs = [f.replace(".jpg", ".png") for f in list_frames]
    list_labels = read_list_labels(os.path.join(labels_path, video))
    n = len(list_segs)
    scores_vid = np.zeros((n-1,2))
    for i in range(1, n):
      pred_seg = Image.open(list_segs[i])
      real_seg = Image.open(list_labels[i])
      pred_seg = np.asarray(pred_seg).reshape(-1)
      real_seg = np.asarray(real_seg).reshape(-1)
      scores_vid[i-1,0] = jaccard_score(real_seg, pred_seg, average='macro')
      scores_vid[i-1,1] = f1_score(real_seg, pred_seg, average='macro')
    scores.append(scores_vid)
  scores = np.concatenate(scores, axis=0)
  scores = np.mean(scores, axis=0)
  print(f'J = {scores[0]}, F = {scores[1]}, J&F = {np.mean(scores)}')
  return scores


