import os
import datetime
from pickle import NONE
from shutil import rmtree

def run_description(DIR, DATASET, MODEL, BATCH_SIZE, N_EPOCHS,LR_MAX, LR_MIN, LOSS, NOTES, SCORE=None):
    f = open(DIR + "model_parameters", "w")
    f.write("Dataset: " + str(DATASET) + "\n")
    f.write("Model: " + str(MODEL) + "\n")
    f.write("Batch size: " + str(BATCH_SIZE) + "\n")
    f.write("Number of epochs: " + str(N_EPOCHS) + "\n")
    f.write("Leaning Rate: " + str(LR_MAX) + "->" + str(LR_MIN) + "\n")
    f.write("Loss: " + LOSS + "\n")
    f.write("Notes: " + str(NOTES) + "\n")
    if SCORE is not None:
        f.write("Test score: " + str(SCORE) + "\n")
    f.close()
    
def create_run_dir_with_description(RUNS_DIR, AUG_PARAMETERS, DATASET_DICT, MODEL_FEATURES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MIN_LEARNING_RATE, LOSS_FUNCTION, LOSS_EXTRA_PARAMS, THRESHOLD):
    RUN_NAME =  str(datetime.datetime.now()) + "/"
    RUN_DIR = RUNS_DIR + RUN_NAME
    os.makedirs(RUN_DIR)
    IMGS_SAVED_DIR = RUN_DIR + "saved_images/"
    os.makedirs(IMGS_SAVED_DIR)
    os.makedirs(IMGS_SAVED_DIR + "val/")
    os.makedirs(IMGS_SAVED_DIR + "test/") 
    EXTRA = {'AUGMENTATION' : AUG_PARAMETERS, 'Threshold' : THRESHOLD}
    if LOSS_EXTRA_PARAMS:
        EXTRA['loss params'] = LOSS_EXTRA_PARAMS
    
    run_description(RUN_DIR, DATASET_DICT, MODEL_FEATURES, BATCH_SIZE, NUM_EPOCHS,LEARNING_RATE, MIN_LEARNING_RATE, LOSS_FUNCTION, NOTES=EXTRA)
    return RUN_DIR

def remove_run_dir(path):
    rmtree(path + 'plot/')
    rmtree(path + 'saved_images/val')
    rmtree(path + 'saved_images/test')
    rmtree(path + 'saved_images/')
    rmtree(path)