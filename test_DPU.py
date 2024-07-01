import os
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as trans
from utils import *
from Base_DPU_model import AutoEncoder
from collections import OrderedDict

height = 256
width = 256
timeStep = 5
numChannels = 3
prototypeSize = 10
testPath = './data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

testDataset = TestDataLoader(testPath, trans.Compose([
                            trans.ToTensor(),
                        ]), height = height, width = width, timeStep = timeStep-1)

videoLoader = data.DataLoader(testDataset, batch_size = 1, shuffle = False, num_workers = 8, drop_last=False)

mseLoss = torch.nn.MSELoss(reduction='none')

model = AutoEncoder(numChannels, timeStep, prototypeSize, 128, 128)
# model.cuda()

dataset = 'ped2'
labels = np.load('./data/frame_labels_' + dataset + '.npy') # anomaly labels

videos = OrderedDict()
videosList = sorted(glob.glob(os.path.join(testPath, '*')))

for video in videosList:
    videoName = video.split('/')[-1]
    videos[videoName] = {}
    videos[videoName]['path'] = video
    videos[videoName]['frame'] = glob.glob(os.path.join(video, '*.tif'))
    videos[videoName]['frame'].sort()
    videos[videoName]['length'] = len(videos[videoName]['frame'])

labelsList = []
anomalyScoreTot = []
labelLength = 0
peakSig2Noise = []
featureDist = []

print('Evaluating model on', dataset)

modelPath = './'

# for video in sorted(videosList):
#     videoName = video.split('/')[-1]
#     videos[videoName]['labels'] = labels[0][4+labelLength:videos[videoName]['length']+labelLength]
#     labelsList = np.append(labelsList, labels[0][stepSize+])
