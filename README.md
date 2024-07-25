# Object-Dection-With-RetinaNet-AS

# Introduction 

Object detection a very important problem in computer vision. Here the model is tasked with localizing the objects present in an image, and at the same time, classifying them into different categories.

Object detection models can be broadly classified into "single-stage" and "two-stage" detectors. Two-stage detectors are often more accurate but at the cost of being slower.

RetinaNet uses a feature pyramid network to efficiently detect objects at multiple scales and introduces a new loss, the Focal loss function, to alleviate the problem of the extreme foreground-background class imbalance.

# Requirements 
* Python 3
* Tensorflow
* Numpy
* Pandas
* Matplotlib
* Keras
* Tensorflow Datasets

# Instructions

# Import Libraries
``
import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

``


