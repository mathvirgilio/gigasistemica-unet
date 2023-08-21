import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from Circular_UNET import UNET
from run import train_fn, validate
from config import *
from dataset import create_loader
from utils import load_checkpoint
from loss import IoULoss, FocalLoss
from torchvision.ops import sigmoid_focal_loss

def main():
    IN_CHANNELS = 1
    #Definição do modelo
    model = UNET(in_channels=IN_CHANNELS, out_channels=1, features = MODEL_FEATURES).to(DEVICE)
    loss_fn = sigmoid_focal_loss #Loss de treinamento
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #Otimizador
    #Carregamento do dataset
    train_loader, val_loader = create_loader(DATASET_DICT, AUG_PARAMETERS, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, in_channels = IN_CHANNELS)
    scaler = torch.cuda.amp.GradScaler()
    
    if LOAD_RUN:
        NAME_CHECKPOINT = '/mnt/data/matheusvirgilio/gigasistemica/UNET/runs/Teste Batch Size = 8 (300 épocas)/checkpoint.pth.tar'
        load_checkpoint(torch.load(NAME_CHECKPOINT, map_location=torch.device(DEVICE)), model)
    if TRAINING:
        train_fn(train_loader, model, optimizer, loss_fn, scaler, val_loader)

    validate(val_loader, model, loss_fn)
    
if __name__ == '__main__':
    main()
    
#Apagar diretório run se ele não possuir um checkpoint
    
    