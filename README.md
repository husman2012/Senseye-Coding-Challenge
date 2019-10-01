# Senseye Coding Challenge

## Introduction
This repository is my solution of the problem presented by Senseye, which was to create a program capable of producing masks given a set pictures of eyes. My solution was to train a UNet model based off resnet18 to create a image segmentation program which accepts images as inputs and returns .png mask images as output. 

## Contents of Repo
This repository contains the following files and folders:
* Solution Files
* Solution Notebook.ipynb
* create_masks.py
* environment.yml
* functions.py
* global_vars.py
* train.py

All of these are required for the application to function. The .py files are required for the command-line application and usage will be explained below. The solution notebook is a Jupyter Notebook explaining the process to the end solution and the Solution Files is necessary as it is the structure the application expects to function correctly. The Jupyter Notebook is meant to show the process and is not guaranteed to work due to that reason.

Within Solution_Files are the following: 
* imgs_small
* input_folder
* masks_small
* output_folder
* codes.txt

Each folder contains a single image that may be deleted. The application expects training files to be located in imgs_small and masks_small however the names may be changed as long as the proper arguments are supplied when calling the training script. The same goes for input_folder and output_folder, along with the same for the arguments for the create_masks.py script. The usage will be explained below.

##Setting up the environment
This application requires FastAI, and other common machine learning libraries. The environment.yml file is provided to allow anyone to setup the environment and run the program. To setup the environment, run the following in anaconda:

```
conda fastai create -f environment.yml
```

Activate the new environment:

```
conda activate fastai
```

## Usage
### train.py
This script is capable of taking several arguments, namely: directory, epochs, learning rate, masks directory, images directory, and batch size. One important note, if the masks and images directories have different names, they must be present in the main directory, similar to how imgs_small and masks_small are present in the Solution_file directory. 
This script requires no arguments except for the Solution_file directory, which may be renamed as long as the argument is supplied but must have the same file structure. To run the file, ensure the terminal is in the directory and run:
```
python train.py Solution_files
```
Example of running with all arguments specified:
```
python train.py diff_directory --learning_rate 0.001 --bs 3 --epochs 2 --images diff_images --masks diff_masks
```
The terminal will show an interface with training progress, as well as output training loss, validation loss, accuracy and time per epoch.

### create_masks.py
This script takes the following arguments: directory, input_folder and output_folder. It will take all images present in the specified input folder (by default: input_folder), create masks for every image and save them in the specified output folder (by default: output folder). To run the script, ensure the terminal is in the correct directory and run:
```
python create_masks.py Solution_files
```
Example of running with all arguments specified:
```
python create_masks.py diff_directory --pred_dir diff_input_folder --save_dir diff_output_folder
```

## Notes
The other files are simply utilities for the scripts. Technically speaking, this command-line application could be applied to any image segmentation problem based on the way it is written. Nothing about the given dataset has been hardcoded except for the directories it expects and codes.txt file which determines the number of classifiable sections in the provided images.
