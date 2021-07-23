import os
import glob
import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from keras.preprocessing import sequence, image
from keras.preprocessing.image import array_to_img, save_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer

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
from keras.models import Model

from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter
from keras import metrics
from evaluation_metrics.multilabel.example_based import (
    hamming_loss,
    example_based_accuracy,
    example_based_precision,
    example_based_recall,
)

from evaluation_metrics.multilabel.label_based import (
    accuracy_macro,
    precision_macro,
    recall_macro,
    accuracy_micro,
    precision_micro,
    recall_micro,
)

from evaluation_metrics.multilabel.alpha_score import alpha_score
from data_loading.load_data import get_tile_data
