import torchvision
from torchvision import torch 
from torch.utils import data
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import pandas as pd
from collections import namedtuple
import random
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
from matplotlib.pyplot import imread
from numpy import asarray
from torchvision.transforms import ToTensor, ToPILImage
import imageio
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from sklearn import metrics
import torch.nn.functional as fun
from statistics import mean 

means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
n_class    = 34

class CityScapesDataset(Dataset):

    def __init__(self, csv_file, n_class=n_class, transforms=None,img_size=512):
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class
        # Add any transformations here
        self.transforms = transforms
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        self.tf = torchvision.transforms.Compose(
            [
                # add more trasnformations as you see fit
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.data)
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        
    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        
        img = Image.open(img_name).convert('RGB')
        
        label_name = self.data.iloc[idx, 1]
        label      = Image.open(label_name)
        
        # call function transform
        im, lbl = self.transform(img, label)
    
        return im, torch.clamp(lbl, max=20)
    
    def transform(self, img, lbl):
       #print('in transform')
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))
            #print('image and label resized to be square')
        
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0
        return img, lbl

            
    def get_labels(self):
        return np.asarray(
            [
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                    [111,74,0],
                    [81,0,81],
                    [128,64,128],
                    [244,35,232],
                    [250,170,160],
                    [230,150,140],
                    [70, 70, 70],
                    [102,102,156],
                    [190,153,153],
                    [180,165,180],
                    [150,100,100],
                    [150,120, 90],
                    [153,153,153],
                    [153,153,153],
                    [250,170,30],
                    [220,220,0],
                    [107,142, 35],
                    [152,251,152],
                    [70,130,180],
                    [220, 20, 60],
                    [255,0,0],
                    [0,0,142],
                    [0,0,70],
                    [0,60,100],
                    [0,0,90],
                    [0,0,110],
                    [0,80,100],
                    [0,0,230],
                    [119,11,32],
            
            ])
        
        
            
    # takes the output of MODEL and puts a seg mask on it
        
    def decode_segmap(self, label_mask, plot=False):
        #print('in decode segmap')
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        
        for ll in range(0, self.n_class):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
            
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
            
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

# defining the dataSets and the dataLoaders
train_dataset = CityScapesDataset(csv_file='/datasets/train_edited.csv')

val_dataset = CityScapesDataset(csv_file='/datasets/val.csv')

test_dataset = CityScapesDataset(csv_file='/datasets/test.csv')


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=True, 

                          )

val_loader = DataLoader(dataset=val_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=False)


test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=False)

#up conv block
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

##########################################################################################################################
# recurrent block
class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        
        self.t = t
        
        self.ch_out = ch_out
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

 ######################################################################################################################

 #RRCNN block
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

 ###############################################################################################################

 # model R2Unet goes here (MAIN MODEL)
class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,n_class,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



dev_id = 7
# change dev_id where we load model as well !
CUDA = torch.cuda.is_available()

CUDA = False

if CUDA:
    print("cuda is available")
    model = R2U_Net().cuda(dev_id)
    
else:
    
    print("cuda not available, calling model on CPU")
    model = R2U_Net()

# loss function
loss_f = nn.CrossEntropyLoss().cuda(dev_id)

# optimizer variable
opt = optim.Adam(model.parameters(), lr=0.001)

TRAIN = False # we set to false as out model is already trained
epochs = 100 # model initially trained on 100 epochs

if TRAIN: 
    print("we are training!")
    
    best_loss = sys.maxsize
    
    for epoch in range(epochs):
      current_loss=0
      for i, d in enumerate(train_loader):
       
        X = d[0] # first element of list is tensor
        y = d[1] # second element is the label
     
        opt.zero_grad()
        
        if CUDA:
            X = X.cuda(dev_id)
            y = y.cuda(dev_id)


        output = model(X) # calling the forward function in NN class

        loss = loss_f(output,y)
       
        current_loss += loss.float()

        loss.backward()
        opt.step()
        
      # we save the model only for the last epoch
      mean_loss = current_loss/len(train_loader)
      if(mean_loss<best_loss):
          torch.save(model.state_dict(),"best.pt")
      print(f"Epoch: {epoch}. Loss: {loss}")

# load the trained model

if CUDA:
    model.load_state_dict(torch.load("best.pt",map_location=torch.device('cuda:7')))
    print("model is loaded for visualization on gpu !")
else:
    model.load_state_dict(torch.load("best.pt",map_location=torch.device('cpu')))

    print("model is loaded for visualization on cpu !")




#### VISUALIZATION PART ################

images = []
labels = []
prediction = []

iterator = enumerate(val_loader)

for i in range(5):
    
    index, d = iterator.__next__()
    
    X = d[0] # original_image
    y = d[1] #label
    
    opt.zero_grad()
    
    if CUDA:
            X = X.cuda(dev_id)
            y = y.cuda(dev_id)
    
    
    model_output_image = model(X)
    
    ##### coverting tensors to numpy arrays for plotting purpose###
    
    nm = torch.argmax(model_output_image, dim=1) # removes channel arg
    
    
    #converts the output image to a numpy array
    n = nm.cpu().numpy()   # makes a numpy array and loads on CPU
    
    # appends output of decode_segmap to list[prediction]
    # decode_segmap takes  a 2d array not a 3d array hence n[0] and not n
    prediction.append(train_dataset.decode_segmap(n[0]))
    
    
    #takes the input image 
    #print(X.shape)    # X gives array of shape (batch_size * channels * H *W) 
    #print(X[0].shape) # X[0] gives array shape (channels*H*W)
    
    xr = X[0].permute(1,2,0).cpu() # puts data on CPU
    xr = np.array(xr) # makes NP array
    #print('xr is:',xr)
    
    # appends this image numpy array to list[images]
    images.append(xr)
    
    
    # takes the label which is a tensor , converts it into a numpy array n2
    n2 = y.cpu().numpy()[0] # makes NP array and computes on CPU

    labels.append(train_dataset.decode_segmap(n2))
    
   
    

 ##### plotting part #######
for i in range(5):
        figs, axs = plt.subplots(nrows=1, ncols=3, figsize =(12,4))
        current = [images[i],prediction[i],labels[i]]
        
        for j,ax in enumerate(axs.flatten()):
            plt.sca(ax)
            plt.imshow(current[j])
        

    

        plt.suptitle('Image | Prediction | Label')
        plt.show()




"""
############ EVALUATION SCRIPT ###################

ID = 3  
CUDA = torch.cuda.is_available()

def load_model(epoch, best_model = False):
    #CUDA =  False
    if CUDA:
        if best_model:
            model.load_state_dict(torch.load('best.pt',map_location=torch.device('cuda:3')))
            print("model loaded in GPU")
            #return model.load_state_dict(torch.load('models/epoch-{}.pt'.format(epoch),map_location=torch.device(ID)))
    
    #if best_model:
        #model.load_state_dict(torch.load('best.pt',map_location=torch.device('cpu')))
        #print("model loaded in CPU")
        #return model.load_state_dict(torch.load('models/epoch-{}.pt'.format(epoch),map_location=torch.device('cpu')))



def get_image_eval(prediction, label):
    pred = torch.flatten(prediction, start_dim = 0, end_dim = 1).cpu().numpy()
    lbl = torch.flatten(label, start_dim = 0, end_dim = 1).cpu().numpy()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    '''
    Confusion matrix contains True positives on the Diagoanl (TP for class i is at M_(i,i)),
    False positives in the column (FP for class i is in column i, except the value at M_(i,i))
    False negatives in the row (FN for class i is in row i, except the value at M_(i,i))
    True Negatives are the rest
    '''
    
    
    conf_matrix = metrics.confusion_matrix(lbl, pred)
    FP = conf_matrix[:].sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix[:].sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix[:].sum() - (FP+FN+TP)
    
    return TP.sum(), TN.sum(), FP.sum(), FN.sum()
    
def get_metrics(prediction, label):
    pred = torch.flatten(prediction, start_dim = 0, end_dim = 1).cpu().numpy()
    lbl = torch.flatten(label, start_dim = 0, end_dim = 1).cpu().numpy()
    f1_score = metrics.f1_score(lbl, pred,average = 'micro')
    JS = metrics.jaccard_score(lbl,pred, average = 'weighted')
    
    return f1_score, JS


if __name__ == "__main__":
    print("calling model R2U-net for evaluation")
    
    
    if CUDA:
        print("cuda is available for evaluation script")
        model = R2U_Net().cuda(ID)
      
    else:
        #model = R2U_Net()
        print("model not called on CUDA !")
    

   
    
    ######### CALL the FUNCTION LOAD MODEL #######################
    load_model(0, best_model = True)
    #############################################################
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    F1 = 0
    JS = 0
    total = 0
    
    for i, d in enumerate(val_loader):
        inp = d[0]
       # print(d[1])
        target = (d[1]).squeeze(dim = 1).long()
        

        if CUDA:
            inp = inp.cuda(ID)
            target = target.cuda(ID)
        
        
        #print("giving input to model")
        pred = model(inp)
        #print(pred.shape)
        
        pred_mask = torch.argmax(pred, dim = 1)
        
        TP_temp, TN_temp, FP_temp, FN_temp = get_image_eval(pred_mask[0], target[0])
        
        #TP_temp2, TN_temp2, FP_temp2, FN_temp2 = get_image_eval(pred_mask[1], target[1])
        
        F1_temp, JS_temp = get_metrics(pred_mask[0], target[0])
        
        #F1_temp2, JS_temp2 = get_metrics(pred_mask[1], target[1])
        
        F1 += F1_temp
        JS += JS_temp 
        TP += TP_temp 
        TN += TN_temp 
        FP += FP_temp 
        FN += FN_temp 
        total += 1
    
    F1_Score = F1/total
    JS_Score = JS/total
    accuracy = (TP + TN) / (TP+TN+FP+FN)
    sensitivity = (TP) / (TP + FN)
    specificity = (TN) / (TN + FP)
    
    print("accuracy: ",accuracy)
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("F1-Score: ",F1_Score)
    print("Jaccard Similarity: ", JS_Score)
"""

#### PLOTTING PART FOR EVAL METRICS ###################
'''
titles = ['F1-Score', 'AUC-ROC Score', 'Jaccard Coefficient']

figs, axs = plt.subplots(nrows=1, ncols=3, figsize =(12,4))

current = [F1_Score,AUC_Score,Dice_Score]


for j,ax in enumerate(axs.flatten()):
    plt.sca(ax)
    plt.title(titles[j])
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.plot(Epochs,current[j])
plt.show()
'''
