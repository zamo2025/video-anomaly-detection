import os
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as trans
from utils import *
from Base_DPU_model import AutoEncoder

height = 256
width = 256
timeStep = 5
numChannels = 3
prototypeSize = 10
ped2TestPath = './data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

testDataset = TestDataLoader(ped2TestPath, trans.Compose([
                            trans.ToTensor(),
                        ]), height = height, width = width, timeStep = timeStep-1)

videoLoader = data.DataLoader(testDataset, batch_size = 1, shuffle = False, num_workers = 8, drop_last=False)

mseLoss = torch.nn.MSELoss(reduction='none')

model = AutoEncoder(numChannels, timeStep, prototypeSize, 128, 128)

