import os
import torch
import torch.utils.data
import torchvision.transforms as trans
from utils import *
from Base_DPU_model import AutoEncoder
from collections import OrderedDict
from tqdm import tqdm
import time

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
model.cuda()

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

print("VideoasList:", videosList)

labelsList = []
anomalyScoreTot = []
labelLength = 0
peakSig2Noise = {}
featureDist = {}

print('Evaluating model on', dataset)

modelPath = './model.pth'

for video in sorted(videosList):
    videoName = video.split('/')[-1]
    videos[videoName]['labels'] = labels[0][4+labelLength:videos[videoName]['length']+labelLength]
    labelsList = np.append(labelsList, labels[0][timeStep-1+labelLength:videos[videoName]['length']+labelLength])
    labelLength += videos[videoName]['length']
    peakSig2Noise[videoName] = []
    featureDist[videoName] = []

model = torch.load(modelPath)
if type(model) is dict:
    model = model['state_dict']
model.cuda()
model.eval()

forward = AverageTracker()
videoNum = 0
updateWeights = None
imgsK = []
kIter = 0
anomalyScoreTot.clear()

for video in sorted(videosList):
    vidName = video.split('/')[-1]
    peakSig2Noise[vidName].clear()
    featureDist[vidName].clear()

progress = tqdm(total=len(videoLoader),
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]',)

with torch.no_grad():
    for k, (frames) in enumerate(videoLoader):
        hiddenState = None
        frames = frames.cuda()
        
        startTime = time.time()
        outputs, featureLoss = model.forward(frames[:, :3*4], updateWeights, False)
        endTime = time.time()
        
        if k >= len(videoLoader) // 2:
            forward.update(endTime - startTime, 1)

        mseImgs = mseLoss((outputs[:] + 1) / 2, (frames[:, -3:] + 1) / 2)

        mseFeas = featureLoss.mean(-1)
        
        mseFeas = mseFeas.reshape((-1,1,256,256))
        mseImgs = mseImgs.view((mseImgs.shape[0],-1))
        mseImgs = mseImgs.mean(-1)
        mseFeas = mseFeas.view((mseFeas.shape[0],-1))
        mseFeas = mseFeas.mean(-1)

        vid = videoNum

        vdd = 0
        for j in range(len(mseImgs)):
            psnr_score = psnr(mseImgs[j].item())
            fea_score = psnr(mseFeas[j].item())
            peakSig2Noise[videosList[vdd].split('/')[-1]].append(psnr_score)
            featureDist[videosList[vdd].split('/')[-1]].append(fea_score)
            kIter += 1
            if kIter == videos[videosList[videoNum].split('/')[-1]]['length']-timeStep+1:
                videoNum += 1
                update_weights = None
                kIter = 0
                imgsK = []
                hiddenState = None

        print('Epoch:')
        print('Video:', dataset, videosList[vid].split('/')[-1])
        print('AEScore:', peakSig2Noise)
        print('Time', endTime)
            
        progress.update(1)

progress.close()
forward.reset()

# Measuring the abnormality score and the AUC
for video in sorted(videosList):
    video_name = video.split('/')[-1]
    template = calc(15, 2)
    aa = filter(anomaly_score_list(peakSig2Noise[video_name]), template, 15)
    bb = filter(anomaly_score_list(featureDist[video_name]), template, 15)
    anomalyScoreTot += score_sum(aa, bb, 0.5)

totAnomalyScore = np.asarray(anomalyScoreTot)
totAccuracy = 100*AUC(totAnomalyScore, np.expand_dims(1 - labelsList, 0))

print('Total AUC:', totAccuracy)
