import numpy as np
import torch
import torchvision
from augmentations import build_train_augmentations, build_test_augmentations
import augmentations
from dataset import UnetDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from main import writer
ITER = 0
#writer = SummaryWriter()
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from torchmetrics.functional import precision, recall, f1_score

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def dice_score_metric(predictions, targets):
    dice_score = (2 * (predictions * targets).sum()) / ((predictions + targets).sum() + 1e-8)
    return dice_score

def get_metrics(pred, y, apply_sigmoid=True):
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    linear_y = torch.reshape(y>0, (-1,)).type(torch.int)
    linear_pred = torch.reshape(pred, (-1,)).type(torch.float)
    
    precision_val = precision(linear_pred, linear_y, task='binary', average='micro')
    recall_val = recall(linear_pred, linear_y, task='binary', average='micro')
    f1_score_val = f1_score(linear_pred,linear_y, task='binary', average='micro')
    return precision_val, recall_val, f1_score_val


def threshold_image(im,th):
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1
    return thresholded_im

def compute_otsu_criteria(im, th):
    thresholded_im = threshold_image(im,th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1
    if weight1 == 0 or weight0 == 0:
        return np.inf
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0 * var0 + weight1 * var1

def find_best_threshold(im):
    threshold_range = range(np.max(im)+1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    return best_threshold


def get_otsu_threshold_from_tensor(pred, apply_sigmoid = False):
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    
    pred_array = pred.cpu().numpy()
    pred_array = (255*pred_array).astype('uint8')
    best_threshold = find_best_threshold(pred_array)
    best_threshold = best_threshold/255
    
    return best_threshold