import argparse
from fastai.vision import *
import random
import numpy as np
import os

def init(filename):
    global path
    global codes
    global void_code

    path = untar_data(os.getcwd())
    path = path/filename
    codes = np.loadtxt(path/'codes.txt', dtype=str)

    name2id = {v:k for k,v in enumerate(codes)}
    void_code = name2id['Void']
