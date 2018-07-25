
# Membrane prediction in NMJ EM images

This codes borrows from [synapse_pytorch](https://github.com/zudi-lin/synapse_pytorch) for synaptic clefts detection in electron microscopy (EM) images using PyTorch.

It is a 3D U-net with several enhancements: 
- residual block, dilation CNN, soft dice and focal loss
- Change concatenation to summation in the expansion path.
- Support training and testing on multi-GPUs.

----------------------------

## Installation

* Clone this repository : `git clone --recursive https://github.com/zudi-lin/synapse_pytorch.git`
* Download and install [Anaconda](https://www.anaconda.com/download/) (Python 3.6 version).
* Create a conda environment :  `conda env create -f synapse_pytorch/envs/py3_pytorch.yml`

## Dataset

W11 and deeper of NMJ EM data. In seperate masks. Export segment data with seeding point on it.

## Training

For iterative training, first label 50 sections for training and predicting, then do proofreading for more ground truth.

### Command

* Run `train.py`.

```
usage: train.py [-h] [-t TRAIN] [-dn IMG_NAME] [-ln SEG_NAME] [-o OUTPUT]
                [-mi MODEL_INPUT] [-ft FINETUNE] [-pm PRE_MODEL] [-lr LR]
                [--volume-total VOLUME_TOTAL] [--volume-save VOLUME_SAVE]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE]

Training Synapse Detection Model

optional arguments:
  -h, --help                Show this help message and exit
  -t, --train               Input folder
  -dn, --img-name           Image data path
  -ln, --seg-name           Ground-truth label path
  -o, --output              Output path
  -mi, --model-input        I/O size of deep network
  -ft, --finetune           Fine-tune on previous model [Default: False]
  -pm, --pre-model          Pre-trained model path
  -lr                       Learning rate [Default: 0.0001]
  --volume-total            Total number of iterations
  --volume-save             Number of iterations to save
  -g, --num-gpu             Number of GPUs
  -c, --num-cpu             Number of CPUs
  -b, --batch-size          Batch size
```

The script supports training on datasets from multiple directories. Make sure that the input dimension is in *zyx*.

### Visulazation
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).

## Prediction

* Run `test.py`.

```
usage: test.py  [-h] [-t TRAIN] [-dn IMG_NAME] [-o OUTPUT] [-mi MODEL_INPUT]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE] [-m MODEL]

Testing Synapse Detection Model

optional arguments:
  -h, --help                Show this help message and exit
  -t, --train               Input folder
  -dn, --img-name           Image data path
  -o, --output              Output path
  -mi, --model-input        I/O size of deep network
  -g, --num-gpu             Number of GPUs
  -c, --num-cpu             Number of CPUs
  -b, --batch-size          Batch size
  -m, --model               Model path used for test
```
