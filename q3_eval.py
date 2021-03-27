
import torch
import torch.nn as nn
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as fun
import time

from q3_train import SegNet_ASPP
from torchvision import transforms
from PIL import Image
from torchvision import datasets
from sklearn import metrics

def load_model(epoch, best_model = False, ):
    if CUDA:
        if best_model:
            model.load_state_dict(torch.load('models/best.pt',map_location=torch.device(ID)))
            return
        model.load_state_dict(torch.load('models/epoch-{}.pt'.format(epoch),map_location=torch.device(ID)))
    
    if best_model:
        model.load_state_dict(torch.load('models/best.pt',map_location=torch.device('cpu')))
        return
    model.load_state_dict(torch.load('models/epoch-{}.pt'.format(epoch),map_location=torch.device('cpu')))

def get_image_eval(prediction, label):
    pred = torch.flatten(prediction, start_dim = 0, end_dim = 1).cpu().numpy()
    lbl = torch.flatten(label, start_dim = 0, end_dim = 1).cpu().numpy()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    """
    Confusion matrix contains True positives on the Diagoanl (TP for class i is at M_(i,i)),
    False positives in the column (FP for class i is in column i, except the value at M_(i,i))
    False negatives in the row (FN for class i is in row i, except the value at M_(i,i))
    True Negatives are the rest
    """
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
    model = SegNet_ASPP()
    CUDA = False
    ID = 0
    if CUDA:
        torch.cuda.empty_cache()
        model = SegNet_ASPP().cuda(ID)
    else:
        model = SegNet_ASPP()
    
    path = "/datasets"
    spl = "val"
    im_tf = transforms.Compose([
        transforms.Resize((256,512)),
        transforms.ToTensor()

    ])
    targ_tf = transforms.Compose([
        transforms.Resize((256,512),interpolation = Image.NEAREST),
        transforms.ToTensor()
    ])
    
    load_model(0, best_model = True)
    dst = datasets.Cityscapes(path, split = spl, mode = 'fine', target_type = 'semantic',transform = im_tf, target_transform = targ_tf)      

    trainloader = torch.utils.data.DataLoader(dataset = dst, batch_size = 2, shuffle = False, num_workers = 0)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    F1 = 0
    JS = 0
    total = 0
    for i, d in enumerate(trainloader):
        inp = d[0]
        target = (d[1]*255).squeeze(dim = 1).long()
        if CUDA:
            inp = inp.cuda(ID)
            target = target.cuda(ID)

        pred = model(inp)
        pred_mask = torch.argmax(pred, dim = 1)
        TP_temp, TN_temp, FP_temp, FN_temp = get_image_eval(pred_mask[0], target[0])
        TP_temp2, TN_temp2, FP_temp2, FN_temp2 = get_image_eval(pred_mask[1], target[1])

        F1_temp, JS_temp = get_metrics(pred_mask[0], target[0])
        F1_temp2, JS_temp2 = get_metrics(pred_mask[1], target[1])
        
        F1 += F1_temp + F1_temp2
        JS += JS_temp + JS_temp2
        TP += TP_temp + TP_temp2
        TN += TN_temp + TN_temp2
        FP += FP_temp + FP_temp2
        FN += FN_temp + FN_temp2
        total += 2

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
