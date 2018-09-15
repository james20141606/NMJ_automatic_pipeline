import json
import os
import time
import tensorflow as tf
import numpy as np
from skimage import io
from scipy.misc import imread, imsave
import imageio
import matplotlib.pyplot as plt
from shutil import copy
from tqdm import tqdm_notebook as tqdm

from AxonDeepSeg.train_network import train_model
from AxonDeepSeg.apply_model import axon_segmentation


# reset the tensorflow graph for new training
tf.reset_default_graph()

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth =True

modelsname = ['2018-08-27_01-06-08','2018-08-24_14-25-36','2018-08-27_01-09-42']
mips = ['mip1','mip2','mip3','mip4']

import sys,argparse,os,time
parser = argparse.ArgumentParser(description='test some parameters')
parser.add_argument('--mip', dest='miplevel',  type=int,default=3, help='mip level')
parser.add_argument('--modelind', dest='modelind',default=0,type=int, help='model ind')
parser.add_argument('--acreso', dest='acreso',default=0.004,type=float, help='acquired resolution')
parser.add_argument('--rereso', dest='rereso',default=0.004,type=float, help='resampled resolution')
args = parser.parse_args()

def predict_loop(path_img ,modelsname,acquired_resolution=0.004,resampled_resolutions=0.004):
    path_folder, file_name = os.path.split(path_img)
    if not os.path.exists(path_folder+'/prediction/'):
        os.makedirs(path_folder+'/prediction/')
    if os.path.isfile(path_folder+'/prediction/'+file_name.split('.')[0]+'_prediction.png'):
        print ('exists, skip')
    else:
        tf.reset_default_graph()
        model_name = 'models/'+modelsname
        path_model = os.path.join(model_name)
        path_configfile = os.path.join(path_model,'config_network.json')
        with open(path_configfile, 'r') as fd:
            config_network = json.loads(fd.read())
        prediction = axon_segmentation(path_folder, file_name, path_model, config_network,
                                       verbosity_level=3, resampled_resolutions = resampled_resolutions, acquired_resolution = acquired_resolution)
        print (path_folder+'/prediction/'+file_name.split('.')[0]+'_prediction.png')
        imsave(path_folder+'/prediction/'+file_name.split('.')[0]+'_prediction.png',prediction[0])

miplevel = args.miplevel
modelind = args.modelind
acreso = args.acreso
rereso = args.rereso

for i in tqdm(os.listdir('data/test/mask_final/'+mips[miplevel])):
    try:
        predict_loop('data/test/mask_final/'+mips[miplevel]+i,modelsname= modelsname[modelind],acquired_resolution=acreso,resampled_resolutions=rereso)
    except:
        print ('file wrong')
