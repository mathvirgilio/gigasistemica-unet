import torch
import os
import datetime

def create_run_dir_with_description():
    RUN_NAME =  str(datetime.datetime.now()) + "/"
    RUN_DIR = RUNS_DIR + RUN_NAME
    os.makedirs(RUN_DIR)
    IMGS_SAVED_DIR = RUN_DIR + "saved_images/"
    os.makedirs(IMGS_SAVED_DIR)
    os.makedirs(IMGS_SAVED_DIR + "val/")
    os.makedirs(IMGS_SAVED_DIR + "test/")
    
    f = open(RUN_DIR + "model_parameters", "w")
    f.write("Model: " + str(MODEL_FEATURES) + "\n")
    f.write("Batch size: " + str(BATCH_SIZE) + "\n")
    f.write("Number of epochs: " + str(NUM_EPOCHS) + "\n")
    f.write("Leaning Rate: " + str(LEARNING_RATE) + "->" + str(MIN_LEARNING_RATE) + "\n")
    f.write("Loss: " + LOSS_FUNCTION + "\n")
    f.write("Notes: " + str(NOTES) + "\n")
    f.close()
    
    return RUN_DIR

# Hyperparameters etc.
MODEL_FEATURES = [32,64,128,256]
LEARNING_RATE = 1e-5
MIN_LEARNING_RATE = 1e-5
BATCH_SIZE = 1
NUM_EPOCHS = 100
LOSS_FUNCTION = "Focal Loss"
APPLY_TTA = False
LOSS_EXTRA_PARAMS = {'one factor' : 25, "apply TTA" : APPLY_TTA}
#LOSS_EXTRA_PARAMS = None
THRESHOLD = 0.5
AUG_PARAMETERS = {"ROTATE" : (0, 0), "HORIZONTAL_FLIP" : 0.2, "VERTICAL_FLIP" : 0.2} #Augmentation
NUM_WORKERS = 0 #2
PIN_MEMORY = True
USE_DATA_PARALLEL = False
#DEVICE = 'cuda' if torch.cuda.is_available() else "cpu" #"cpu"
DEVICE = 'cuda:1' if torch.cuda.is_available() else "cpu"
#DEVICE = 'cpu'
IN_CHANNELS = 1
LOAD_RUN = False
TRAINING = True

#Folder paths
DATASET_DIR = "/mnt/data/matheusvirgilio/gigasistemica/dataset_quarter"
TRAIN_IMG_DIR = "/mnt/data/matheusvirgilio/gigasistemica/dataset_quarter/images/train/"
TRAIN_MASK_DIR = "/mnt/data/matheusvirgilio/gigasistemica/dataset_quarter/masks/train/"
VAL_IMG_DIR = "/mnt/data/matheusvirgilio/gigasistemica/dataset_quarter/images/val/"
VAL_MASK_DIR = "/mnt/data/matheusvirgilio/gigasistemica/dataset_quarter/masks/val/"
DATASET_DICT = {"train_images"  : TRAIN_IMG_DIR, "train_masks" : TRAIN_MASK_DIR, 
                "val_images"    : VAL_IMG_DIR,   "val_masks"   : VAL_MASK_DIR,}
# Description
NOTES = 'Describe run'
RUNS_DIR = "/mnt/data/matheusvirgilio/gigasistemica/UNET/runs/"
RUN_DIR = create_run_dir_with_description()
IMG_PATH = RUN_DIR + "saved_images/"
NAME_CHECKPOINT = RUN_DIR + "checkpoint.pth.tar"