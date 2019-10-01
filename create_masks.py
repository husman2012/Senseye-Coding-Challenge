import global_vars
import functions
import argparse
from fastai.vision import *
import numpy as np
import os

#Produce final masks, save into folder specified
def acc (input, target):
        target = target.squeeze(1)
        mask = target != global_vars.void_code
        return (input.argmax(dim = 1)[mask] == target[mask]).float().mean()

parser = argparse.ArgumentParser(description = "train")

parser.add_argument('filename', type = str)
parser.add_argument('--pred_dir', default = 'input_folder')
parser.add_argument('--save_dir', default = 'output_folder')

args = parser.parse_args()
global_vars.init(args.filename)

#Create path variables
path_img = global_vars.path/args.pred_dir
path_lbl = global_vars.path/args.save_dir

#Create list of image file names
fnames = get_image_files(path_img)

#Load learner
learn = load_learner(global_vars.path)

#Function to get names of images to later save masks with the same name
get_y_fn = lambda x: path_lbl/f'{x.stem}.png'

n_masks = len(fnames)
for image in fnames:
    img = open_image(image)
    y = learn.predict(img)[0]
    y.save((get_y_fn(image)))

#Condition check if user didn't put any masks in correct folder
if n_masks == 0:
    print("No images found in {}".format(args.pred_dir))
print("{} masks saved in {}".format(n_masks, args.save_dir))
