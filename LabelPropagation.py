import numpy as np
from PIL import Image
import cv2
from glob import glob
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import queue
from tqdm import tqdm
from urllib.request import urlopen

def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

def read_frame(frame_path, patch_size=16, dino=False, ori_shape=False):
  frame = cv2.imread(frame_path)
  h, w, c = frame.shape
  if ori_shape:
    new_w = int((w//patch_size)*patch_size)
    new_h = h
  else:
    new_w = 224
    new_h = 224

  frame = cv2.resize(frame, (new_w, new_h))

  frame = frame.astype(np.float32)
  frame = frame / 255.0
  if dino:
    frame = frame[:,:,::-1]
  frame = np.transpose(frame.copy(), (2, 0, 1))
  frame = torch.from_numpy(frame).float()
  # usual vits normalize frames
  if dino:
    frame = color_normalize(frame)

  return frame, h, w

def read_label(label_path, patch_size=16, ori_shape=False):
  labels = Image.open(label_path)
  w, h = labels.size
  # Make sure that all images can be divided into patches
  if ori_shape :
    new_w = int((w//patch_size)*patch_size)
    new_h = h
  else :
    new_w = 224
    new_h = 224
  # Resize so it is the same size than our encoded image
  resized_labels = labels.resize((new_w // patch_size, new_h // patch_size), 0)
  # display(resized_labels.resize((w,h),0))
  resized_labels = np.array(resized_labels)
  # cv2_imshow(resized_labels)
  resized_labels = torch.from_numpy(resized_labels.copy())
  # One hot encoding
  w, h = resized_labels.shape
  # Number of dims of one hot encoding correspond to the max int in the array
  n_dims = int(resized_labels.max()+1)
  flattened_labels = resized_labels.type(torch.LongTensor).view(-1,1)
  one_hot = torch.zeros((flattened_labels.shape[0], n_dims)).scatter(1, flattened_labels, 1)
  one_hot = one_hot.view(h, w, n_dims)
  return one_hot.permute(2,0,1).unsqueeze(0), np.asarray(labels)


def read_list_frames(video_path):
  return sorted([file for file in glob(os.path.join(video_path, '*.jpg'))])

def read_list_labels(labels_path):
  return sorted([file for file in glob(os.path.join(labels_path, '*.png'))])


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask


def compute_mask(h, w, size_neighborhood):
  mask = torch.zeros(h, w, h, w)
  for i in range(h):
    for j in range(w):
      for p in range(2 * size_neighborhood + 1):
        for q in range(2 * size_neighborhood + 1):
          if i - size_neighborhood + p < 0 or i - size_neighborhood + p >= h:
            continue
          if j - size_neighborhood + q < 0 or j - size_neighborhood + q >= w:
            continue
          mask[i, j, i - size_neighborhood + p, j - size_neighborhood + q] = 1

  mask = mask.reshape(h * w, h * w)
  return mask.cuda(non_blocking=True)

def encode_frame(model, frame, patch_size=16, dino=False):
  if dino:
    out = model(frame.unsqueeze(0)).last_hidden_state[0].unsqueeze(0)
    out = out[:, 1:, :]
    h, w = int(frame.shape[1] / patch_size), int(frame.shape[2] / patch_size)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)

  else:
    frame = frame.unsqueeze(0).cuda()
    h, w = int(frame.shape[2] / patch_size), int(frame.shape[3] / patch_size)
    encoded_test_data =  model.forward_encoder(frame, 0)
    out = encoded_test_data[:,1:,:].squeeze()
  return out, h, w


def label_propagation(model, list_past_features, plabels, tframe, Tau, k, size_neighborhood=0, dino=False):
  tfeature, h, w = encode_frame(model, tframe, dino=dino)
  m = len(list_past_features)
  tfeatures = tfeature.cuda().unsqueeze(0).repeat(m, 1, 1)
  pfeatures = torch.stack(list_past_features).cuda()
  tfeatures = F.normalize(tfeatures, dim=1, p=2)
  pfeatures = F.normalize(pfeatures, dim=1, p=2)

  aff = torch.exp(torch.bmm(tfeatures, pfeatures.permute(0, 2, 1))/Tau) # produce the dot product between target frame and m context frames
  if size_neighborhood > 0:
    mask = compute_mask(h, w, size_neighborhood).unsqueeze(0).repeat(m, 1, 1)
    aff *= mask

  aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
  tk_val, _ = torch.topk(aff, dim=0, k=k)
  tk_val_min, _ = torch.min(tk_val, dim=0)
  aff[aff < tk_val_min] = 0

  aff = aff / torch.sum(aff, keepdim=True, axis=0)

  list_segs = [s.cuda() for s in plabels]
  segs = torch.cat(list_segs)
  nmb_context, C, h, w = segs.shape
  segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
  tseg = torch.mm(segs, aff)
  tseg = tseg.reshape(1, C, h, w)

  return tseg, tfeature


def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')




@torch.no_grad()
def eval_davis(model, video_name, videos_path, labels_path, m, model_name, patch_size=16, dino=False):
  color_palette = []

  for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
    color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
  color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)



  list_frames = read_list_frames(os.path.join(videos_path, video_name))
  list_labels = read_list_labels(os.path.join(labels_path, video_name))

  first_frame, ori_h, ori_w = read_frame(list_frames[0], dino=dino)
  first_frame_seg, seg_ori = read_label(list_labels[0])

  first_frame_feat, _ , _ = encode_frame(model, first_frame, dino=dino)
  que = queue.Queue(m)
  for i in tqdm(range(1, len(list_frames))):
    past_frames_feats = [first_frame_feat] + [pair[0]  for pair in que.queue]
    past_frames_segs = [first_frame_seg] + [pair[1] for pair in que.queue]

    target_frame = read_frame(list_frames[i], dino=dino)[0]
    frame_seg, frame_feature = label_propagation(model, past_frames_feats, past_frames_segs, target_frame, 1, 7, 10, dino=dino)

    if que.qsize() == m:
      que.get()
    que.put([frame_feature, frame_seg])

    frame_seg = F.interpolate(frame_seg, scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
    frame_seg = norm_mask(frame_seg)
    _, frame_seg = torch.max(frame_seg, dim=0)
    frame_seg = np.array(frame_seg.squeeze().cpu(), dtype=np.uint8)
    frame_seg = np.array(Image.fromarray(frame_seg).resize((ori_w, ori_h), 0))
    frame_nm = list_frames[i].split('/')[-1].replace(".jpg", ".png")
    
    imwrite_indexed(os.path.join(videos_path, model_name, video_name, frame_nm), frame_seg, color_palette)