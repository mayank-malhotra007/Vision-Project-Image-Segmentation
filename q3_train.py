import os
import sys
from os.path import join
import collections
import json
import torch
import torch.nn as nn
import imageio
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import torch.nn as nn
import torch.nn.functional as fun
import time

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from matplotlib.pyplot import imread


class Segnet_Encoder(nn.Module):
    def __init__(self):
        super(Segnet_Encoder, self).__init__()
        #define the layers for your model
        #Stage 1
        self.enc_conv1a = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
        self.enc_conv1b = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        
        #Stage 2
        self.enc_conv2a = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.enc_conv2b = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        
        #Stage 3
        self.enc_conv3a = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.enc_conv3b = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.enc_conv3c = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)

        
        #Stage 4
        self.enc_conv4a = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
        self.enc_conv4b = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.enc_conv4c = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        
        #Stage 5
        self.enc_conv5a = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.enc_conv5b = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.enc_conv5c = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        
        
        
        #decoder part

        #Stage 5
        self.dec_conv5c = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.dec_conv5b = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.dec_conv5a = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)

        #Stage 4
        self.dec_conv4c = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.dec_conv4b = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
        self.dec_conv4a = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 3, padding = 1)
        
        #Stage 3
        self.dec_conv3c = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.dec_conv3b = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.dec_conv3a = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1)
        
        #Stage 2
        self.dec_conv2b = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.dec_conv2a = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1)
        
        #Stage 1
        self.dec_conv1b = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)


        
        #batch normalization
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_64  = nn.BatchNorm2d(64)
        

        
    def forward(self, x):
        #define the forward pass
        fun = nn.functional
        s_0 = x.size()      
        #encoder pass
         
        #Stage 1
        x = fun.relu(self.bn_64(self.enc_conv1a(x)))
        x = fun.relu(self.bn_64(self.enc_conv1b(x)))
        x, i_1 = fun.max_pool2d(x, kernel_size = 2, stride = 2, return_indices = True)
        s_1 = x.size()
    
        #Stage 2
        x = fun.relu(self.bn_128(self.enc_conv2a(x)))
        x = fun.relu(self.bn_128(self.enc_conv2b(x)))
        x, i_2 = fun.max_pool2d(x, kernel_size = 2, stride = 2, return_indices = True)
        s_2 = x.size()
        
        #Stage 3
        x = fun.relu(self.bn_256(self.enc_conv3a(x)))
        x = fun.relu(self.bn_256(self.enc_conv3b(x)))
        x = fun.relu(self.bn_256(self.enc_conv3c(x)))
        x, i_3 = fun.max_pool2d(x, kernel_size = 2, stride = 2, return_indices = True)
        s_3 = x.size()
    
        #Stage 4
        x = fun.relu(self.bn_512(self.enc_conv4a(x)))
        x = fun.relu(self.bn_512(self.enc_conv4b(x)))
        x = fun.relu(self.bn_512(self.enc_conv4c(x)))
        x, i_4 = fun.max_pool2d(x, kernel_size = 2, stride = 2, return_indices = True)
        s_4 = x.size()
    
        #Stage 5
        x = fun.relu(self.bn_512(self.enc_conv5a(x)))
        x = fun.relu(self.bn_512(self.enc_conv5b(x)))
        x = fun.relu(self.bn_512(self.enc_conv5c(x)))
        x, i_5 = fun.max_pool2d(x, kernel_size = 2, stride = 2, return_indices = True)
        #decoder pass
        
        #Stage 5
        x = fun.max_unpool2d(x,i_5,kernel_size = 2, stride = 2, output_size = s_4)
        x = fun.relu(self.bn_512(self.dec_conv5c(x)))
        x = fun.relu(self.bn_512(self.dec_conv5b(x)))
        x = fun.relu(self.bn_512(self.dec_conv5a(x)))

        #Stage 4
        x = fun.max_unpool2d(x,i_4,kernel_size = 2, stride = 2, output_size = s_3)
        x = fun.relu(self.bn_512(self.dec_conv4c(x)))
        x = fun.relu(self.bn_512(self.dec_conv4b(x)))
        x = fun.relu(self.bn_256(self.dec_conv4a(x)))
        
        #Stage 3
        x = fun.max_unpool2d(x,i_3,kernel_size = 2, stride = 2, output_size = s_2)
        x = fun.relu(self.bn_256(self.dec_conv3c(x)))
        x = fun.relu(self.bn_256(self.dec_conv3b(x)))
        x = fun.relu(self.bn_128(self.dec_conv3a(x)))
        
        #Stage 2
        x = fun.max_unpool2d(x,i_2,kernel_size = 2, stride = 2, output_size = s_1)
        x = fun.relu(self.bn_128(self.dec_conv2b(x)))
        x = fun.relu(self.bn_64(self.dec_conv2a(x)))

        #Stage 1
        x = fun.max_unpool2d(x,i_1,kernel_size = 2, stride = 2, output_size = s_0)
        x = fun.relu(self.bn_64(self.dec_conv1b(x)))
        
        return x


class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        self.pyramide_layers = nn.ModuleList()
        self.initialize_Pyramide()
    
    def initialize_Pyramide(self):
        rates = [6, 12, 18]
        in_ch = 64
        out_ch = 256
        conv = nn.Sequential(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = 1),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU())
        self.pyramide_layers.append(conv)
        for rate in rates:
            current = nn.Sequential(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, dilation = rate, padding = rate),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU())
            self.pyramide_layers.append(current)
        pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = 1),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU())
        self.pyramide_layers.append(pooling)
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels = len(self.pyramide_layers)*out_ch, out_channels = out_ch, kernel_size = 1),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(),
                                        nn.Dropout(0.5))
        
    def forward(self, x):
        concatenation = []
        for layer in self.pyramide_layers:
            y = layer(x)
            concatenation.append(y)
        concatenation[-1] = fun.interpolate(concatenation[-1], size = x.shape[-2:], mode = 'bilinear')
        
        concatenated = torch.cat(concatenation, dim = 1)
        final = self.final_conv(concatenated)
        
        return final


class SegNet_ASPP(nn.Module):
    """
    Combined model
    """
    def __init__(self):
        super(SegNet_ASPP, self).__init__()
        self.encoder = Segnet_Encoder()
        self.pyramide = ASPP()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 35, kernel_size = 1)
        
    def forward(self, x):
        y = self.encoder(x)
        y = self.pyramide(y)
        y = self.conv1(y)
        y = self.conv2(y)
        
        return y


def init_weights(m, VGG16_WEIGHTS = True):
    """
    Initializes the weights and biases.
    If VGG16_WEIGHTS is true, the SegNet encoder will be initialized with a pretrained VGG16 models weights and biases.
    Default is to use the VGG16 weiggst and biases
    """
    def randomize(mod):
        
        if isinstance(mod,nn.Conv2d):
            torch.nn.init.xavier_normal_(mod.weight.data)
            if(mod.bias != None):
                torch.nn.init.zeros_(mod.bias.data)
    m.apply(randomize)
    m.encoder.apply(randomize)
    m.pyramide.apply(randomize)
    if VGG16_WEIGHTS:
        vgg16 = models.vgg16(pretrained = True).cuda(ID)
        m.encoder.enc_conv1a.weight.data = vgg16.features[0].weight.data
        m.encoder.enc_conv1a.bias.data   = vgg16.features[0].bias.data

        m.encoder.enc_conv1b.weight.data = vgg16.features[2].weight.data
        m.encoder.enc_conv1b.bias.data   = vgg16.features[2].bias.data

        m.encoder.enc_conv2a.weight.data = vgg16.features[5].weight.data
        m.encoder.enc_conv2a.bias.data   = vgg16.features[5].bias.data

        m.encoder.enc_conv2b.weight.data = vgg16.features[7].weight.data
        m.encoder.enc_conv2b.bias.data   = vgg16.features[7].bias.data

        m.encoder.enc_conv3a.weight.data = vgg16.features[10].weight.data
        m.encoder.enc_conv3a.bias.data   = vgg16.features[10].bias.data

        m.encoder.enc_conv3b.weight.data = vgg16.features[12].weight.data
        m.encoder.enc_conv3b.bias.data   = vgg16.features[12].bias.data

        m.encoder.enc_conv3c.weight.data = vgg16.features[14].weight.data
        m.encoder.enc_conv3c.bias.data   = vgg16.features[14].bias.data

        m.encoder.enc_conv4a.weight.data = vgg16.features[17].weight.data
        m.encoder.enc_conv4a.bias.data   = vgg16.features[17].bias.data

        m.encoder.enc_conv4b.weight.data = vgg16.features[19].weight.data
        m.encoder.enc_conv4b.bias.data   = vgg16.features[19].bias.data

        m.encoder.enc_conv4c.weight.data = vgg16.features[21].weight.data
        m.encoder.enc_conv4c.bias.data   = vgg16.features[21].bias.data

        m.encoder.enc_conv5a.weight.data = vgg16.features[24].weight.data
        m.encoder.enc_conv5a.bias.data   = vgg16.features[24].bias.data

        m.encoder.enc_conv5b.weight.data = vgg16.features[26].weight.data
        m.encoder.enc_conv5b.bias.data   = vgg16.features[26].bias.data

        m.encoder.enc_conv5c.weight.data = vgg16.features[28].weight.data
        m.encoder.enc_conv5c.bias.data   = vgg16.features[28].bias.data


def get_class_probabilities(self):
    """
    Computes class probabilities for a weighted Cross-Entropy Loss
    However we decided not to use it for the final training.
    """
    probs = dict((i,0) for i in range(35))
    for im_name in self.files[self.split]:
        label_path = pjoin(self.root, "SegmentationClass/pre_encoded", im_name + ".png")
        raw = Image.open(label_path).resize((224,224))
        img_np = np.array(raw).reshape(224*224)
        img_np[img_np==255] = self.n_classes-1
        for i in range(self.n_classes):
            probs[i] += np.sum(img_np == i)
    values = np.array(list(probs.values()))
    p_values = values/np.sum(values)

    return torch.Tensor(p_values)


if __name__ == "__main__":
    CUDA = torch.cuda.is_available()
    TRAIN = True
    bs = 3
    epochs = 100
    ID = 3
    path = "/datasets"
    spl = "train"
    im_tf = transforms.Compose([
        transforms.Resize((256,512)),
        transforms.ToTensor()

    ])
    targ_tf = transforms.Compose([
        transforms.Resize((256,512),interpolation = Image.NEAREST),
        transforms.ToTensor()
    ])

    dst = datasets.Cityscapes(path, split = spl, mode = 'fine', target_type = 'semantic',transform = im_tf, target_transform = targ_tf)      

    trainloader = torch.utils.data.DataLoader(dataset = dst, batch_size = bs, shuffle = True, num_workers = 0)
    if CUDA:
        torch.cuda.empty_cache()
        model = SegNet_ASPP().cuda(ID)
    else:
        model = SegNet_ASPP()


    if TRAIN:

        init_weights(model)
        model.load_state_dict(torch.load('models/best.pt',map_location=torch.device(ID)))
        if CUDA:
            loss_f = nn.CrossEntropyLoss().cuda(ID)
        else:
            loss_f = nn.CrossEntropyLoss()

        opt = torch.optim.Adam(model.parameters())
        loss_best = sys.maxsize
        dir = "models/"
        for epoch in range(93,epochs):
            print("Epoch:", epoch)
            current_loss = 0
            b = 350
            start = time.time()
            for i, d in enumerate(trainloader):
                inp = d[0]
                if(inp.size() == (bs,3,256,512)):
                    target = (d[1]*255).squeeze().long()
                    if CUDA:
                        inp = inp.cuda(ID)
                        target = target.cuda(ID) 
         
                    pred = model(inp)
                    opt.zero_grad()
                    loss = loss_f(pred,target)
                    loss.backward()
                    opt.step()
                    current_loss += loss.float()
                    if(i%350 == 349 and i!=0): 
                        print("Current Loss",current_loss/b)
                        b += 350
                else:
                    print(inp.size())
                    print(i)
            end = time.time()
            print("Epoch runtime: ",end-start," seconds")
           # torch.save(model.state_dict(), os.path.join(dir, 'epoch-{}.pt'.format(epoch)))      
           """uncommment the previous line if weights should be saved each epoch
           """
            if current_loss<loss_best:
                """
                Saves the model with the lowest per-epoch loss
                """
                loss_best = current_loss
                torch.save(model.state_dict(),os.path.join(dir,'best.pt'))



