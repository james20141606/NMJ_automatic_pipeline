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

from AxonDeepSeg.train_network import train_model
from AxonDeepSeg.apply_model import axon_segmentation

# reset the tensorflow graph for new training
tf.reset_default_graph()

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True

dir_name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")
# dir_name = 'the_name_of_my_model_folder'

path_model = os.path.join('models/', dir_name)

if not os.path.exists(path_model):
    os.makedirs(path_model)

filename = '/config_network.json'

parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
parser.add_argument('-t','--train',  default='/n/coxfs01/',
                    help='Input folder (train)')
parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                    help='Image data path')
parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                    help='Ground-truth label path')
parser.add_argument('-o','--output', default='result/train/',
                    help='Output path')
parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                    help='I/O size of deep network')
args = parser.parse_args()


trainingset_name = 'mip1'
path_training = 'data/train/mip1/trainingpath'

config = {
    
# General parameters:    
  "n_classes": 3,
  "thresholds": [0, 0.2, 0.8],    
  "trainingset_patchsize": 512,    
  "trainingset": "mip1",    
  "batch_size": 8,

# Network architecture parameters:     
  "depth": 2,
  "convolution_per_layer": [2, 2],
  "size_of_convolutions_per_layer": [[3, 3], [3, 3]],
  "features_per_convolution": [[[1, 5], [5, 5]], [[5, 10], [10, 10]]],
  "downsampling": "maxpooling",
  "dropout": 0.75,
     
# Learning rate parameters:    
  "learning_rate": 0.001,    
  "learning_rate_decay_activate": True,    
  "learning_rate_decay_period": 24000, 
  "learning_rate_decay_type": "polynomial", 
  "learning_rate_decay_rate": 0.99,
    
# Batch normalization parameters:     
  "batch_norm_activate": True,     
  "batch_norm_decay_decay_activate": True,    
  "batch_norm_decay_starting_decay": 0.7, 
  "batch_norm_decay_ending_decay": 0.9, 
  "batch_norm_decay_decay_period": 16000,
        
# Weighted cost parameters:    
  "weighted_cost-activate": True, 
  "weighted_cost-balanced_activate": True, 
  "weighted_cost-balanced_weights": [1.1, 1, 1.3], 
  "weighted_cost-boundaries_sigma": 2, 
  "weighted_cost-boundaries_activate": False, 
    
# Data augmentation parameters:    
  "da-type": "all", 
  "da-2-random_rotation-activate": False, 
  "da-5-noise_addition-activate": False, 
  "da-3-elastic-activate": True, 
  "da-0-shifting-activate": True, 
  "da-4-flipping-activate": True, 
  "da-1-rescaling-activate": False    
}

if os.path.exists(path_model+filename):
    with open(path_model+filename, 'r') as fd:
        config_network = json.loads(fd.read())
else: # There is no config file for the moment
    with open(path_model+filename, 'w') as f:
        json.dump(config, f, indent=2)
    with open(path_model+filename, 'r') as fd:
        config_network = json.loads(fd.read())

tf.reset_default_graph()

train_model(path_training, path_model, config)
