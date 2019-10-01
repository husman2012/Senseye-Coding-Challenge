import global_vars
import functions
import argparse
from fastai.vision import *
import random
import numpy as np
import os


def main():
    
    #Create argument parser and parse arguments
    parser = argparse.ArgumentParser(description = "train")
    parser.add_argument('filename', type = str)
    parser.add_argument('--images', default = "imgs_small")
    parser.add_argument('--masks', default = "masks_small")
    parser.add_argument('--learning_rate', default = 0.002, type = float)
    parser.add_argument('--epochs', default = 1, type = int)
    parser.add_argument('--bs', default = 2, type = int)
    args = parser.parse_args()
    
    #Initialize global variables
    global_vars.init(args.filename)
    
    #Create path variables
    path = global_vars.path
    path_lbl = path/args.masks
    path_img = path/args.images
    
    #Create data object
    data = functions.create_data(path, args.bs, path_lbl, path_img)
    
    #Create Learner Object
    learn = unet_learner(data, models.resnet18, metrics = functions.acc, wd = 1e-2)
    print("\nBeginning training with {} epochs, batch size of {}, and learning rate of {}\n"
          .format(args.epochs, args.bs, args.learning_rate))
    
    #Begin training based on specified arguments
    learn.fit_one_cycle(args.epochs, args.learning_rate)
    
    #Save model in directory
    learn.export()
    print("Model saved as export.pkl in {}".format(args.filename))

if __name__ == '__main__':
    main()
