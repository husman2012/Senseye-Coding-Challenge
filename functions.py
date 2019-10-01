import global_vars
from fastai.vision import *
import random
import numpy as np
import os

#Metrics variables

def acc (input, target):
    '''
    Metrics function to assess accuracy of model
    input: input, target
    returns: accuracy in float of all pixel values
    '''
    
    target = target.squeeze(1)
    mask = target != global_vars.void_code
    return (input.argmax(dim = 1)[mask] == target[mask]).float().mean()


def create_data(path, bs, path_lbl, path_img):
    """
    Function to create DataBunch object from FastAI
    inputs:
        path, WindowsPathObject
        bs, batch_size int
        path_lbl, WindowsPathObject
        path_img, WindowsPathObject
    returns:
        data, DataBunch Object from FastAI
    """
    
    #Create lists for image and label files
    fnames = get_image_files(path_img)
    lbl_names = get_image_files(path_lbl)

    #function required to get paths to each image in masks
    get_y_fn = lambda x: path_lbl/f'{x.stem}.png'

    #Function to open respective masks based on original image
    img_f = fnames[10]
    mask = open_mask(get_y_fn(img_f))
    
    #Get size of mask and reduce
    src_size = np.array(mask.shape[1:])
    size = src_size//2
    
    #create validation list
    create_segments(path_img)

    #Create DataBunch
    src = (SegmentationItemList.from_folder(path_img).split_by_fname_file('../valid.txt').label_from_func(get_y_fn, classes=global_vars.codes))
    data = (src.transform(get_transforms(), size=size, tfm_y=True).databunch(bs=bs, path = path).normalize(imagenet_stats))

    return data

#Function to randomly create validation list ("valid.txt")
def create_segments(path_to_img, pct = 0.10):
    """
    Function to create valid.txt to randomly select validation set.
    10% is the default value
    
    arguments: WindowsPath Object to img folder
    returns: New file named valid.txt into path directory
    """
    
    mask_list = os.listdir(path_to_img)
    num_files = int(len(mask_list)*pct)
    valid_list = random.sample(mask_list, num_files)
    outF = open(path_to_img/"../valid.txt", "w")
    for line in valid_list:
        outF.write(line)
        outF.write("\n")
    outF.close()





