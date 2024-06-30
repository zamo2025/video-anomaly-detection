import os
import numpy as np
import cv2
import torch
import torchvision as tv
import torchvision.transforms as trans
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from Base_DPU_model import AutoEncoder
from utils import *


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



batchSize = 4
frameReconstuctLoss = 1.0
featReconstuctLoss = 1.0
disLossRate = 0.0001
# ped1 = np.load('./data/frame_labels_ped1.npy', allow_pickle=True)

ped2TrainPath = './data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
# ped2TestPath = './data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

ped2TrainVideos = extract_videos(ped2TrainPath)
print(ped2TrainVideos.shape)

ped2Labels = np.load('./data/frame_labels_ped2.npy', allow_pickle=True)
                                  
videoTensors = torch.tensor(ped2TrainVideos, dtype=torch.float32)
print(type(videoTensors))
dataset = TensorDataset(videoTensors)
print(type(dataset))
videoLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
print(type(videoLoader.dataset))

# Train the base model
model = AutoEncoder()
# model.cuda()

modelParams = []
modelParams.extend(model.encoder.parameters())
modelParams.extend(model.decoder.parameters())
modelParams.extend(model.prototype.parameters())
modelParams.extend(model.outhead.parameters())

# Optimize model parameters
optimizer = torch.optim.Adam(modelParams, lr=1e-4)

startEpoch = 0
model = torch.nn.DataParallel(model)

mseLoss = torch.nn.MSELoss(reduction='none')
pixelLoss = AverageTracker()
featureLoss = AverageTracker()
distinguishedLoss = AverageTracker()

model.train()

numEpochs = 10
for epoch in range(startEpoch, numEpochs):
    labels = []

    # Shows the progress of the training
    progress = tqdm(total=len(videoLoader))

    for i, (frames) in enumerate(videoLoader):
        framesTensor = frames[0]
        framesTensor = framesTensor.view(batchSize, -1, framesTensor.shape[-2], framesTensor.shape[-1])

        print(framesTensor.shape)

        output, _, _, _, featLoss, _, disLoss, = model.forward(framesTensor[:,0:12], None, True)
        optimizer.zero_grad()
        pixelLoss = torch.mean(mseLoss(output, framesTensor[:,12:]))
        featLoss = featLoss.mean()
        disLoss = disLoss.mean()
        loss = frameReconstuctLoss*pixelLoss + featReconstuctLoss*featLoss + disLossRate*disLoss
        loss.backward(retain_graph=True)
        optimizer.step()

# for batch in videoLoader:
#     vbatch = batch[0]
#     firstVideo = vbatch[0]
#     firstFrame = firstVideo[0]
#     frame = firstFrame.numpy().astype(np.uint8)
#     plt.imshow(frame)
#     plt.show()
