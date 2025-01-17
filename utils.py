import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import random
import pickle
import math
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(42)

def psnr(mse):
    return 10 * math.log10(1 / mse)

def calc(r=5, sigma=2):
    k = np.zeros(r)
    for i in range(r):
        k[i] = 1/((2*math.pi)**0.5*sigma)*math.exp(-((i-r//2)**2/2/(sigma**2)))
    return k

def anomaly_score(psnr, maxPSNR, minPSNR):
    return ((psnr - minPSNR) / (maxPSNR - minPSNR))

def anomaly_score_list(psnrList):
    anomalyScoreList = list()
    for i in range(len(psnrList)):
        anomalyScoreList.append(anomaly_score(psnrList[i], np.max(psnrList), np.min(psnrList)))
        
    return anomalyScoreList

def score_sum(list1, list2, alpha):
    result = []
    for i in range(len(list1)):
        result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return result

def AUC(anomalyScores, labels):
    frameAUC = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomalyScores))
    return frameAUC

# Normalize the images from [0, 255] to [0, 1]
def np_load_frame(filename, height, width):
    decodedImg = cv2.imread(filename)
    resizedImg = cv2.resize(decodedImg, (width, height))
    resizedImg = resizedImg.astype(dtype=np.float32)
    resizedImg = (resizedImg / 127.5) - 1.0
    return resizedImg

# Extends Dataset, specifically used to manage the video data during for the testing process
class TestDataLoader(data.Dataset):
    def __init__(self, folder, transform, height, width, timeStep=4, numPred=1):
        self.directory = folder
        self.transform = transform
        self.videos = OrderedDict()
        self.height = height
        self.width = width
        self.timeStep = timeStep
        self.numPred = numPred
        self.setup()
        self.samples = self.get_all_samples()
         

    def setup(self):
        videos = glob.glob(os.path.join(self.directory, '*'))
        for video in sorted(videos):
            videoName = video.split('/')[-1]
            self.videos[videoName] = {}
            self.videos[videoName]['path'] = video
            self.videos[videoName]['frame'] = glob.glob(os.path.join(video, '*.tif'))
            self.videos[videoName]['frame'].sort()
            self.videos[videoName]['length'] = len(self.videos[videoName]['frame'])
                   
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.directory, '*'))
        for video in sorted(videos):
            videoName = video.split('/')[-1]
            for i in range(len(self.videos[videoName]['frame']) - self.timeStep):
                frames.append(self.videos[videoName]['frame'][i])
                           
        return frames               
               
    def __getitem__(self, index):
        videoName = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        batch = []
        for i in range(self.timeStep+self.numPred):
            print(type(self.videos[videoName]['frame']))
            image = np_load_frame(self.videos[videoName]['frame'][frame_name+i], self.height, self.width)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
            
    def __len__(self):
        return len(self.samples)

# Extention of the Dataset, specifically used to prepare the train videos for training
class TrainDataLoader(data.Dataset):
    def __init__(self, directory, dataset, transform, height, width, timeStep=4, segments=32, numPred=1, batchSize=1):
        self.directory = directory
        self.dataset = dataset
        self.transform = transform
        self.videos = OrderedDict()
        self.videoNames = []
        self.height = height
        self.width = width
        self.timeStep = timeStep
        self.numPred = numPred
        self.num_segs = segments
        self.batchSize = batchSize
        self.setup()
        
    def setup(self):
        folder = self.directory
        file = './data/frame_' + self.dataset + '.pickle'

        if os.path.exists(file):
            file = open(file,'rb')
            self.videos = pickle.load(file)
            for name in self.videos:
                self.videoNames.append(name)
        else:
            videos = glob.glob(os.path.join(folder, '*'))
            
            for video in sorted(videos):
                videoName = video.split('/')[-1]
                self.videoNames.append(videoName)
                self.videos[videoName] = {}
                self.videos[videoName]['path'] = video
                self.videos[videoName]['frame'] = glob.glob(os.path.join(video, '*.tif'))
                self.videos[videoName]['frame'].sort()
                self.videos[videoName]['length'] = len(self.videos[videoName]['frame'])
            
    def __getitem__(self, index):
        
        videoName = self.videoNames[index]
        length = self.videos[videoName]['length']-self.timeStep
        seg_ind = random.sample(range(0, self.num_segs), self.batchSize)
        frame_ind = random.sample(range(0, length//self.num_segs), 1)

        batch = []
        for j in range(self.batchSize):
            frame_name = seg_ind[j]*(length//self.num_segs)+frame_ind[0]
        
            for i in range(self.timeStep+self.numPred):
                image = np_load_frame(self.videos[videoName]['frame'][frame_name+i], self.height, self.width)
                if self.transform is not None:
                    batch.append(self.transform(image))
        return np.concatenate(batch, axis=0)
    
    def __len__(self):
        return len(self.videoNames)

# Class that is used to track the average of some quantity  
class AverageTracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.average = 0
        self.count = 0
        self.sum = 0
        self.value = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count