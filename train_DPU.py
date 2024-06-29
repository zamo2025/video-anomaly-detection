import os
import numpy as np
import cv2
import torch
import torchvision as tv
import torchvision.transforms as trans
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# Ensure each frame is 256 by 256 and each video have 200 frames
def pad_videos(videos):
    paddedVideos = []
    for video in videos:
        paddedVideo = np.zeros((200, 256, 256, 3), dtype=np.uint8)
        paddedVideos.append(paddedVideo)

    return np.array(paddedVideos)

def extract_videos(path):
    videos = []
    for videoFolder in sorted(os.listdir(path)):
        if videoFolder.startswith('Train'):
            frames = []
            currentFolder = path + '/' + videoFolder
            for filename in sorted(os.listdir(currentFolder)):
                if filename.endswith('.tif'):
                    imagePath = os.path.join(currentFolder, filename)
                    frame = cv2.imread(imagePath)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            videos.append(frames)

    paddedVideos = pad_videos(videos)

    return paddedVideos



# ped1 = np.load('./data/frame_labels_ped1.npy', allow_pickle=True)

ped2TrainPath = './data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
# ped2TestPath = './data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

ped2TrainVideos = extract_videos(ped2TrainPath)
print(ped2TrainVideos.shape)

ped2Labels = np.load('./data/frame_labels_ped2.npy', allow_pickle=True)
                                  
videoTensors = torch.tensor(ped2TrainVideos, dtype=torch.float32)
dataset = TensorDataset(videoTensors)
videoLoader = DataLoader(dataset, batch_size=4, shuffle=True)

# for batch in videoLoader:
#     vbatch = batch[0]
#     firstVideo = vbatch[0]
#     firstFrame = firstVideo[0]
#     frame = firstFrame.numpy().astype(np.uint8)
#     plt.imshow(frame)
#     plt.show()
