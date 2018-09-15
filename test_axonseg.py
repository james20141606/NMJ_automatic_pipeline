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

parser = argparse.ArgumentParser(
    description='Training Synapse Detection Model')
parser.add_argument('-g', '--gpu',  default='4/',
                    help='gpu to use')
parser.add_argument('-dn', '--img-name',  default='im_uint8.h5',
                    help='Image data path')
parser.add_argument('-ln', '--seg-name',  default='seg-groundtruth2-malis.h5',
                    help='Ground-truth label path')
parser.add_argument('-o', '--output', default='result/train/',
                    help='Output path')
parser.add_argument('-mi', '--model-input', type=str,  default='31,204,204',
                    help='I/O size of deep network')
args = parser.parse_args()

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True


path_img = args.img


path_model = os.path.join('models/', dir_name)

if not os.path.exists(path_model):
    os.makedirs(path_model)

filename = '/config_network.json'
trainingset_name = 'mip1'
path_training = 'data/train/mip1/trainingpath'

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
