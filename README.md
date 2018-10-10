# Automatic segmentation for NMJ

Also see my [website page](https://www.cmwonderland.com/blog/2018/09/12/100_summer_intern/)


![Markdown](http://i2.tiimg.com/640680/584c64fdaf11c64e.png)

This work is inspired from yaron and marco’s great work on automatically prediction membrane on bundle. And we are thinking, if we only care about tracing, maybe we can automatically trace the axon with little manual label. Since the bundle data is sparse and the shift of the z section is big, we may use a simpler yet more robust way to automaticaly trace.
So at first we will prepare the data, use some methods to generate more, and we will do segment prediction to get a segment and post process it, then use matching algorithm to trace each axon.

#### Data  preparation
- Extract axon segment (from Marco’s data)

![Markdown](http://i2.tiimg.com/640680/9d013476a19d3dd8.png) 

![Markdown](http://i2.tiimg.com/640680/3c2d8c2fe9cab2e7.png)

- Convert all segments to same color
	- Training:	1200
	- Validation: 200
	
![Markdown](http://i2.tiimg.com/640680/9e61c84bcda13810.png)

At first we use KK and marco’s data as training and validation sample. We convert the segment to same color as binary mask

#### Data Augmentation
### Training:
- Simple augmentation: 
	- flip of x, y, (z); 
	- 90 degree rotation.
	
![Markdown](http://i2.tiimg.com/640680/688954a7b2f85b05.png)

- Intensity augmentation.

![Markdown](http://i2.tiimg.com/640680/d7f84ed10dc6977f.png)

- Elastic augmentation

![Markdown](http://i1.fuimg.com/640680/3e78d3e458b063b3.png)

#####  Test:
Simple augmentation(16 combination)

Several augmentation methods are applied here to generate more training data, we have simple augmentation, intensity and elastic augmentation. For test part, we do all kinds of simple augmentation to get the average result
Although the augmentation May not have the strong biological meaning, but it is always useful to optimize the model better.

#### Prediction Model
We have discussed a lot about the prediction model, after a long time’s try, the 2D Dlinknet (adjustmen of U-net) finally works.

3D U-net with res block  (not very good)
2D D-LinkNet: encoder-decoder, res block, dilation.

![Markdown](http://i2.tiimg.com/640680/70f0977b0a1fdfa5.png)


- Loss: 
BCE+DICE loss(It seems remove DICE may have better result)

![Markdown](http://i2.tiimg.com/640680/3b59f2842eaf0b18.png)

![Markdown](http://i2.tiimg.com/640680/713830f2720ff701.png)

Now we use a deep learning model to predict segmentation. I tried 3D U-net and 2D Link net to predict segment. It seems the 2D model is easier to train, for it has less parameters to tune and our data may have a big shift cross z-section. The model is similar to U-net, and the loss function we use is the combination of DICE loss and focal loss, which depict the overlap and difference of ground truth and prediction. The loss function decreases as training goes on.

which adopts encoderdecoder structure, dilated convolution and pretrained encoder, D-LinkNet architecture. Each blue rectangular block represents a multi-channel features map. Part A is the encoder of D-LinkNet. D-LinkNet uses ResNet34 as encoder. Part C is the decoder of D-LinkNet, it is set the same as LinkNet decoder. Original LinkNet only has Part A and Part C. D-LinkNet has an additional Part B which can enlarge the receptive field and as well as preserve the detailed spatial information. Each convolution layer is followed by a ReLU activation except the last convolution layer which use sigmoid activation.

reduces the relative loss for well-classified examples (pt > .5), putting more focus on hard, misclassified examples. (we propose to reshape the loss function to down-weight easy examples and thus focus training on hard negatives. More formally, we propose to add a modulating factor (1 − pt) γ to the cross entropy loss, with tunable focusing parameter γ ≥ 0. We define the focal loss as)

#### Prediction Result
### Post processing:
- Bilateral filter
- Erosion
- Dilation

![Markdown](http://i2.tiimg.com/640680/3ccf827be716fafb.png)

I did some post processing work on prediction, using bilateral filter to remove some noise, Bilateral filter is better than gaussian filter. and use erosion and dilation to remove the potential merge of different connected region, since it is important to get sparse segment for next matching step, the dilation will make the segment smaller than the ground truth.
I evaluate it on validation set and the dice coefficient is acceptable since most of the region overlaps well.
Evaluation on Validation set


![Markdown](http://i2.tiimg.com/640680/e24603eb94d53f89.png)

![Markdown](http://i2.tiimg.com/640680/840d29a47250f731.png)


# Automatically skeletonize and segmentation


Since NMJ project contains a very large volume EM data which has some serious problems to process it automatically(hard to align, image quality is not good, axons travel fast). The project progress seems really slow. There are about 200 NMJs, and we should generate about 200 masks, each mask may contain 300 sections. So the manually seeding and segment work seems really challenging and time-consuming. I am considering to do it more automatically.

# Pipeline
The complete pipeline should contain: 
**Generating Masks —> Seeding —> Predict Membrane —> Expand Seeds —> Merge different Masks**

We would like to build up the whole pipeline, prepare all the codes and model for prediction and processing and write down the protocol.

## Predict Membrane
The automatically prediction parts must include membrane prediction, because it is “easier” to predict since the raw image already have the membrane.

#### training steps
- train loss
![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/trainloss.png)

- visualize output during training(Use TensorboardX)

EM image

![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/em.png)

Ground truth image

![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/gt.png)

Predict image

![](https://github.com/james20141606/Summer_Intern/blob/master/NMJ/plot/predict.png)

It seems the training is quite well after only thousands batches within one hour.
## 3D U-net

<img src="https://github.com/james20141606/Summer_Intern/blob/master/synapse_prediction/plot/Digraph.gv-1.png" style="width: 2px;"/>

## 2D D-Linknet

![](https://github.com/james20141606/NMJ_automatic_pipeline/blob/master/10.png)

##  Automatically seeding
The traditional way is to manually put seeds on each axon, but we have approximately 50,000 sections if all masks are generated, it is so time-consuming to manually put seeds. I will g**enerate seeds by distance transformation from membrane**

Then the seeds must be indexed to track each seed is from which axon, so we will manually put seeds  per 100 sections, then do **Hungarian matching.**

- Merge masks
We are thinking about linear interpolation to merge anchor sections for loop problems.

# Algorithm
## Predict Membrane
Use 3D U-net using contours from dense segmentation sections. Use 50 sections for training, then predict more, proofread predicted sections to generate more training samples. **The iterative training and predicting method will make the model more precise.**
## Automatically seeding
- Distance transformation
- Hungarian matching

This repository is a re-implementation of [Synapse-unet](https://github.com/zudi-lin/synapse-unet) (in Keras) for synaptic clefts detection in electron microscopy (EM) images using PyTorch. However, it contains some enhancements of the original model:

* Add residual blocks to the orginal unet.
* Change concatenation to summation in the expansion path.
* Support training and testing on multi-GPUs.

----------------------------

## Installation

* Clone this repository : `git clone --recursive https://github.com/zudi-lin/synapse_pytorch.git`
* Download and install [Anaconda](https://www.anaconda.com/download/) (Python 3.6 version).
* Create a conda environment :  `conda env create -f synapse_pytorch/envs/py3_pytorch.yml`

## Dataset

Use contours of Dense segmentation labels

## Training

### Command

* Activate previously created conda environment : `source activate ins-seg-pytorch`.
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

The script supports training on datasets from multiple directories. Please make sure that the input dimension is in *zyx*.

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

## Evaluation

Run `evaluation.py -p PREDICTION -g GROUND_TRUTH`.
The evaluation script will count the number of false positive and false negative pixels based on the evaluation metric from [CREMI challenge](https://cremi.org/metrics/). Synaptic clefts IDs are NOT considered in the evaluation matric. The inputs will be converted to binary masks.




