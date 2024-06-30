# This file creates the base auto encoder with the DPU built 
# into the feed forward process.

import torch
from collections import OrderedDict
import torch.nn.functional as func
from dynamic_prototype_unit import *

class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel = 3):
        super(Encoder, self).__init__()

        def BaseConvLayers1(numInput, numOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=numInput, out_channels=numOutput, 
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=numOutput, out_channels=numOutput, 
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        
        def BaseConvLayers2(numInput, numOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=numInput, out_channels=numOutput, 
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=numOutput, out_channels=numOutput, 
                                kernel_size=3, stride=1, padding=1)
            )
        
        self.conv1 = BaseConvLayers1(n_channel*(t_length-1), 64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BaseConvLayers1(64, 128)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BaseConvLayers1(128, 256)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = BaseConvLayers2(256, 512)

    def forward(self, x):
        tensorConv1 = self.conv1(x)
        tensorPool1 = self.pool1(tensorConv1)

        tensorConv2 = self.conv2(tensorPool1)
        tensorPool2 = self.pool2(tensorConv2)

        tensorConv3 = self.conv3(tensorPool2)
        tensorPool3 = self.pool3(tensorConv3)

        tensorConv4 = self.conv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3

class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Decoder, self).__init__()

        def BaseConvLayers(numInput, numOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=numInput, out_channels=numOutput, 
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=numOutput, out_channels=numOutput, 
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
            )
        
        def Deconvolve(numChannels, numOutputs):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=numChannels, out_channels=numOutputs, 
                                        kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.ReLU(inplace=False)
            )
        
        self.conv1 = BaseConvLayers(512, 512)
        self.deconvolve1 = Deconvolve(512, 512)

        self.conv2 = BaseConvLayers(512, 256)
        self.deconvolve2 = Deconvolve(256, 128)

        self.conv3 = BaseConvLayers(256, 128)
        self.deconvolve3 = Deconvolve(128, 64)

    def forward(self, input, skip3, skip2, skip1):
        tensorConv = self.conv1(input)
        tensorDeconvolve1 = self.deconvolve1(tensorConv)
        concat1 = torch.cat((skip1, tensorDeconvolve1), dim=1)

        tensorConv2 = self.conv2(concat1)
        tensorDeconvolve2 = self.deconvolve2(tensorConv2)
        concat2 = torch.cat((skip2, tensorDeconvolve2), dim=1)

        tensorConv3 = self.conv3(concat2)
        tensorDeconvolve3 = self.deconvolve2(tensorConv3)
        concat3 = torch.cat((skip3, tensorDeconvolve3), dim=1)

        return concat3


class AutoEncoder(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=5, proto_size=10, 
                 feature_dim=512, key_dim=512, temp_update=0.1, 
                 temp_gather=0.1):
        super(AutoEncoder, self).__init__()

        def Outputhead(numInput, numOutput, numChannels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=numInput, out_channels=numChannels, 
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=numChannels, out_channels=numOutput, 
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        # TODO: Write the part of the DPU that breaks the encoded normal vectors into prototypes
        self.prototype = Dynamic_Prototype(proto_size, feature_dim, key_dim, temp_update, temp_gather)
        self.outhead = Outputhead(128, n_channel, 64)

    def set_learnable_params(self, layers):
        for name, param in self.named_parameters():
            if any([name.startswith(layer) for layer in layers]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_learnable_parameters(self):
        params = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                params[name] = param

        return params
    
    def get_params(self, layers):
        params = OrderedDict()
        for name, param in self.named_parameters():
            if any([name.startswith(layer) for layer in layers]):
                params[name] = param

        return params
    
    def forward(self, input, weights=None, train=True):
        feature, skip1, skip2, skip3 = self.encoder(input)
        newFeature = self.decoder(feature, skip1, skip2, skip3)
        normNewFeature = func.normalize(newFeature, dim = 1)

        if train:
            updatedFeature, keys, featureLoss, cstLoss, disLoss = self.prototype(normNewFeature, normNewFeature, 
                                                                                 weights, train)
            if weights == None:
                output = self.outhead(updatedFeature)
            else:
                input = func.conv2d(updatedFeature, weights['ohead.0.weight'], weights['ohead.0.bias'], stride=1, padding=1)
                input = func.relu(input)
                input = func.conv2d(input, weights['ohead.2.weight'], weights['ohead.2.weight'], stride=1, padding=1)
                input = func.relu(input)
                input = func.conv2d(input, weights['ohead.4.weight'], weights['ohead.4.weight'], stride=1, padding=1)
                output = func.tanh(input)

            return output, feature, updatedFeature, keys, featureLoss, cstLoss, disLoss
        
        else:
            updatedFeatures, keys, query, featureLoss = self.prototype(normNewFeature, normNewFeature, 
                                                                       weights, train)
            if weights == None:
                output = self.outhead(updatedFeature)
            else:
                input = func.conv2d(updatedFeature, weights['ohead.0.weight'], weights['ohead.0.bias'], stride=1, padding=1)
                input = func.relu(input)
                input = func.conv2d(input, weights['ohead.2.weight'], weights['ohead.2.weight'], stride=1, padding=1)
                input = func.relu(input)
                input = func.conv2d(input, weights['ohead.4.weight'], weights['ohead.4.weight'], stride=1, padding=1)
                output = func.tanh(input)

            return output, featureLoss

# TODO: Write the function that initialies the traning process for the MPU
# def train_init():

# TODO: Write the function that initialies the testing process for the MPU
# def test_init():      