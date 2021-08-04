import pandas as pd
import numpy as np
import pickle
import glob
import json
import os
from collections import Counter
from PIL import Image, ImageOps

from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer

from keras.preprocessing import sequence, image
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import model_from_json
from keras.models import Model
from keras.layers import (
    Flatten,
    Dense,
    Input,
    Activation,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    Dropout,
    UpSampling2D,
    Lambda,
)
from keras.layers import ReLU, Reshape, Conv2DTranspose, Concatenate, Multiply
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping