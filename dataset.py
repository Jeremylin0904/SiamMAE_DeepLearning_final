from tensorflow import keras
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from data_config import *
from tqdm import tqdm
import gc
import os
import json

class SiameseDataset(Dataset):
    def __init__(self, dataset=None, transform=None):
        self.dataset = self.load_data_from_file()
        self.transform = transform

    def load_data_from_file(self, indexes=5):
        dataset = {}
        for index in range(indexes):
            with np.load(f"data/data_{index}.npz", allow_pickle=True) as data:
                file_size = os.path.getsize(f"data/data_{index}.npz")
                print(f"The size of data/data_{index}.npz is: {file_size} bytes")
                loaded_dict = dict(data)
            dataset.update(loaded_dict)
        return dataset


    def init_items(self):
        self.training_vector = []
        self.labels_vector = []
        for key, value in tqdm(self.dataset.items()):
            for v in value:
                self.training_vector.append((v[0], v[1]))
                self.labels_vector.append(1)

            random_idx = np.random.choice(len(list(self.dataset.values())))
            temp_key = list(self.dataset.keys())[random_idx]
            while key == temp_key:
                random_idx = np.random.choice(len(list(self.dataset.values())))
                temp_key = list(self.dataset.keys())[random_idx]
            random_idx_2 = np.random.choice(len(self.dataset[temp_key]))
            self.training_vector.append((self.dataset[key][random_idx_2][0], self.dataset[temp_key][random_idx_2][1]))
            self.labels_vector.append(0)

    def get_frames(self):
        idx = np.random.choice(len(self.training_vector))
        frame_1 = self.training_vector[idx][0]
        frame_2 = self.training_vector[idx][1]
        label = self.labels_vector[idx]
        return frame_1, frame_2, label

    def show_first_and_future_frames(self, frame_1, frame_2, transform=True):
        if transform:
            frame_1 = frame_1.transpose(1, 2, 0)
            frame_2 = frame_2.transpose(1, 2, 0)

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(keras.utils.array_to_img(frame_1))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(keras.utils.array_to_img(frame_2))
        plt.axis("off")
        plt.show()

def load_videos(categories):
    videos = {}
    with open("data/done_indexes.json", "r") as f:
        done_indexes = json.load(f)["done"]

    intervals = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40)]
    for index, (key, value) in tqdm(enumerate(categories.items())):
        if index in done_indexes:
            continue
        if index == 10:
            break
        for video in value[:10]:
            video_vector = load_video(fetch_ucf_video(video))
            for intrv in intervals:
                temp_intrv = np.arange(intrv[0], intrv[1])
                idxs = np.random.permutation(temp_intrv)[:2]
                idxs.sort()
                try:
                    videos[key].append([video_vector[idxs[0]], video_vector[idxs[1]]])
                except KeyError:
                    videos[key] = [[video_vector[idxs[0]], video_vector[idxs[1]]]]

            dir_path = f"data/"
            file_path = os.path.join(dir_path, f"data_{index}")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            np.savez(file_path, **videos)
            # with open(file_path, "wb") as file:
            #     file.write(videos)
            # After processing each video, try to free up memory
            del video_vector
            gc.collect()

    return videos


if __name__ == "__main__":

    categories = show_video_list(show=False)

    videos = load_videos(categories)

    transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])

    #siamese_dataset = SiameseDataset(videos)
    siamese_dataset = SiameseDataset()
    siamese_dataset.init_items()
    frame_1, frame_2, label = siamese_dataset.get_frames()
    if label == 1:
        print("This is a positive pair")
    else:
        print("This is a negative pair")
    siamese_dataset.show_first_and_future_frames(frame_1, frame_2, transform=False)

    
