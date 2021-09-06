import os
import glob
import pandas as pd
import numpy as np
<<<<<<< HEAD
import pickle
=======
>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
from sklearn.utils import shuffle
import json
from keras.preprocessing import sequence, image
from keras.preprocessing.image import array_to_img, save_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import model_from_json
<<<<<<< HEAD
from keras.models import Model
from keras import backend as K
=======
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
>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
from collections import Counter

from utils.evaluation_metrics.multilabel.example_based import (
    hamming_loss,
    example_based_accuracy,
    example_based_precision,
    example_based_recall,
)

from utils.evaluation_metrics.multilabel.label_based import (
    accuracy_macro,
    precision_macro,
    recall_macro,
    accuracy_micro,
    precision_micro,
    recall_micro,
)

from utils.evaluation_metrics.multilabel.alpha_score import alpha_score
from utils.data_loading.load_data import get_tile_data

<<<<<<< HEAD
data_directory = "../data/context_data/"
json_directory = "../data/json_files_trimmed_features/"

# data loading
=======
##loading data
data_directory = "../data/context_data/"
json_directory = "../data/json_files_trimmed_features/"
>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
data = get_tile_data(data_directory, json_directory)
print("\nThe size of total data is", data.shape)
data = shuffle(data)
# split into train-test
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
print("\nThe size of the train data is ", train_data.shape)
print("The size of the test data is ", test_data.shape)

# load the multilabel binarizer
<<<<<<< HEAD
with open("../model/model_tokenizer_test.pickle", "rb") as handle:
=======
import pickle

with open("model_tokenizer_test.pickle", "rb") as handle:
>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
    mlb = pickle.load(handle)
print("Feature Dictionary Loaded")
total_features = len(mlb.classes_)
print("The feature dictionary has size", total_features)
print("Features", mlb.classes_)

# load entire autoencoder architecture
<<<<<<< HEAD
json_file = open("../model/autoencoder_model_test.json", "r")
loaded_model_json = json_file.read()
json_file.close()
ae_sep_output = model_from_json(loaded_model_json)
ae_sep_output.load_weights("../model/autoencoder_model_test.h5")
print("Loaded Entire Autoencoder Model from the Disk")

# load the encoding architecture and weights
json_file = open("../model/encoder_model_test.json", "r")
loaded_model_json = json_file.read()
json_file.close()
encoding_model = model_from_json(loaded_model_json)
encoding_model.load_weights("../model/encoder_model_test.h5")
print("Loaded Encoder Model from the Disk")

# load the decoding architecture and weights
json_file = open("../model/decoder_model_test.json", "r")
loaded_model_json = json_file.read()
json_file.close()
decoding_model = model_from_json(loaded_model_json)
decoding_model.load_weights("../model/decoder_model_test.h5")
=======
json_file = open("autoencoder_model_test.json", "r")
loaded_model_json = json_file.read()
json_file.close()
ae_sep_output = model_from_json(loaded_model_json)
ae_sep_output.load_weights("autoencoder_model_test.h5")
print("Loaded Entire Autoencoder Model from the Disk")

# load the encoding architecture and weights
json_file = open("encoder_model_test.json", "r")
loaded_model_json = json_file.read()
json_file.close()
encoding_model = model_from_json(loaded_model_json)
encoding_model.load_weights("encoder_model_test.h5")
print("Loaded Encoder Model from the Disk")

# load the decoding architecture and weights
json_file = open("decoder_model_test.json", "r")
loaded_model_json = json_file.read()
json_file.close()
decoding_model = model_from_json(loaded_model_json)
# load weights into new model
decoding_model.load_weights("decoder_model_test.h5")
>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
print("Loaded Decoder Model from the Disk")

# Build Input Output Training Batches
print("Building Training Batches")
<<<<<<< HEAD
=======

>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
"""Note : Add Generators"""
train_image_batch = []
for train_path in train_data["image_path"]:
    tile = image.load_img(train_path, target_size=(48, 48))
    tile_sprite = image.img_to_array(tile)
    train_image_batch.append(tile_sprite)
train_image_batch = np.array(train_image_batch)
train_text_batch = []
for i in range(len(train_data["features"])):
    text_ = mlb.transform(train_data["features"][i : i + 1])
    train_text_batch.append(text_)
train_text_batch = np.array(train_text_batch).reshape(
    train_data.shape[0], total_features
)

output_image_batch = []
for i in range(len(train_image_batch)):
    current_image = train_image_batch[i]
    current_image_centre = train_image_batch[i][16 : 16 + 16, 16 : 16 + 16, :]
    output_image_batch.append(current_image_centre)
output_image_batch = np.array(output_image_batch)
output_text_batch = []
for i in range(len(train_text_batch)):
    current_text = train_text_batch[i]
    output_text_batch.append(current_text)
output_text_batch = np.array(output_text_batch)
print("Training Data Ready")
print("Train Image batch shape", train_image_batch.shape)
print("Train Text batch shape", train_text_batch.shape)
print("Output Image batch shape", output_image_batch.shape)
print("Output Text batch shape", output_text_batch.shape)


# Build Input Output Test Batches
print("Building Testing Batches")
"""Note : Add Generators"""
test_image_batch = []
for test_path in test_data["image_path"]:
    tile = image.load_img(test_path, target_size=(48, 48))
    tile_sprite = image.img_to_array(tile)
    test_image_batch.append(tile_sprite)
test_image_batch = np.array(test_image_batch)
test_text_batch = []
for i in range(len(test_data["features"])):
    text_ = mlb.transform(test_data["features"][i : i + 1])
    test_text_batch.append(text_)
test_text_batch = np.array(test_text_batch).reshape(test_data.shape[0], total_features)
print("\n\nTesting Data Ready")
print("Train Image batch shape", test_image_batch.shape)
print("Train Text batch shape", test_text_batch.shape)

predicted_image, predicted_text = ae_sep_output.predict(
    [test_image_batch, test_text_batch]
)
y_pred = [np.where(text > 0.5, 1, 0) for text in predicted_text]
y_pred = np.array(y_pred)
print("Predicted Y is Ready. Shape : ", y_pred.shape)

y_true = test_text_batch
y_true = np.array(y_true)
print("True Y is Ready. Shape :", y_true.shape)

true_image = []
for i in range(len(test_image_batch)):
    current_image = test_image_batch[i]
    current_image_centre = test_image_batch[i][16 : 16 + 16, 16 : 16 + 16, :]
    true_image.append(current_image_centre)
true_image = np.array(true_image)
print("Predicted Array shape ", predicted_image.shape)
print("True Array shape ", true_image.shape)

mse_dist = []
for idx in range(len(true_image)):
    y_true_image = true_image[idx]
    y_true_image = y_true_image.reshape(16, 16, 3)

    y_pred_image = predicted_image[idx]
    y_pred_image = y_pred_image.reshape(16, 16, 3)

    mse_dist.append(np.mean(np.subtract(y_true_image, y_pred_image) ** 2))

print("Mean MSE", np.mean(mse_dist))
print("Median MSE", np.median(mse_dist))

<<<<<<< HEAD
=======

>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
def valid_divide(num, den):
    count = 0
    result = {}
    for idx in range(len(num)):
        if num[idx] == den[idx] == 0:
            continue
        elif num[idx] != 0 and den[idx] != 0:
            result[idx] = num[idx] / den[idx]
            count += 1
        elif num[idx] != 0 and den[idx] == 0 or num[idx] == 0 and den[idx] != 0:
            count += 1
            result[idx] = 0.0
    return result, count


print("\nMacro Label Based Precision", precision_macro(y_true, y_pred))
print("Macro Label Based Recall", recall_macro(y_true, y_pred))
print("Macro Label Based Accuracy", accuracy_macro(y_true, y_pred))

print("\nMicro Label Based Precision", precision_micro(y_true, y_pred))
print("Micro Label Based Recall", recall_micro(y_true, y_pred))
print("Micro Label Based Accuracy", accuracy_micro(y_true, y_pred))

print("\nExample Based Precision", example_based_precision(y_true, y_pred))
print("Example Based Recall", example_based_recall(y_true, y_pred))
print("Example Based Accuracy", example_based_accuracy(y_true, y_pred))