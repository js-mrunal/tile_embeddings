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
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
import tensorflow as tf

=======
>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
##loading train and testing data

data_directory = "../data/context_data/"
json_directory = "../data/json_files_trimmed_features/"
data = get_tile_data(data_directory, json_directory)
print("\nThe size of total data is", data.shape)
data = shuffle(data)
# split into train-test
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
print("\nThe size of the train data is ", train_data.shape)
print("The size of the test data is ", test_data.shape)
# Feature Dictionary
print("Building feature Dictionary..")
mlb = MultiLabelBinarizer()
combined_features = np.concatenate(
    [train_data["features"], test_data["features"]], axis=0
)
mlb_model = mlb.fit(combined_features)
total_features = len(mlb_model.classes_)
print("The feature dictionary has size", total_features)
print("Printing Feature classes")
print(mlb_model.classes_)

# Build Input Output Training Batches
print("Building Training Batches")
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

# model definition
latent_dim = 128
batch_size = 1
# image encoder
image_encoder_input = Input(shape=(48, 48, 3), name="image_input")
image_encoder_conv_layer1 = Conv2D(
    32, strides=3, kernel_size=(3, 3), name="iencode_conv1"
)(image_encoder_input)
image_encoder_norm_layer1 = BatchNormalization()(image_encoder_conv_layer1)
image_encoder_actv_layer1 = ReLU()(image_encoder_norm_layer1)
image_encoder_conv_layer2 = Conv2D(32, (3, 3), padding="same", name="iencode_conv2")(
    image_encoder_actv_layer1
)
image_encoder_norm_layer2 = BatchNormalization()(image_encoder_conv_layer2)
image_encoder_actv_layer2 = ReLU()(image_encoder_norm_layer2)
image_encoder_conv_layer3 = Conv2D(16, (3, 3), padding="same", name="iencode_conv3")(
    image_encoder_actv_layer2
)
image_encoder_norm_layer3 = BatchNormalization()(image_encoder_conv_layer3)
image_encoder_actv_layer3 = ReLU()(image_encoder_norm_layer3)
image_shape_before_flatten = K.int_shape(image_encoder_actv_layer3)[1:]
image_flatten = Flatten(name="image_flatten_layer")(image_encoder_actv_layer3)

# text encoder
text_encoder_input = Input(shape=(13,))
text_encoder_dense_layer1 = Dense(32, activation="tanh", name="tencode_dense1")(
    text_encoder_input
)
text_encoder_dense_layer2 = Dense(16, activation="tanh", name="tencode_dense2")(
    text_encoder_dense_layer1
)
text_shape_before_concat = K.int_shape(text_encoder_dense_layer2)[1:]

# image-text concatenation
image_text_concat = Concatenate(name="image_text_concatenation")(
    [image_flatten, text_encoder_dense_layer2]
)
image_text_concat = Dense(256, activation="tanh", name="embedding_dense_1")(
    image_text_concat
)


# define encoder model
encoding_model = Model(
    inputs=[image_encoder_input, text_encoder_input], outputs=image_text_concat
)

# decoder for image
# decoder_input=Input(shape=(512,))
image_y = Dense(units=np.prod(image_shape_before_flatten), name="image_dense")(
    image_text_concat
)
image_y = Reshape(target_shape=image_shape_before_flatten, name="image_reshape")(
    image_y
)
image_decoder_convt_layer1 = Conv2DTranspose(
    16, (3, 3), padding="same", name="idecode_conv1"
)(image_y)
image_decoder_norm_layer1 = BatchNormalization(name="idecode_norm1")(
    image_decoder_convt_layer1
)
image_decoder_actv_layer1 = ReLU(name="idecode_relu1")(image_decoder_norm_layer1)
image_decoder_convt_layer2 = Conv2DTranspose(
    32, (3, 3), padding="same", name="idecode_conv2"
)(image_decoder_actv_layer1)
image_decoder_norm_layer2 = BatchNormalization(name="idecode_norm2")(
    image_decoder_convt_layer2
)
image_decoder_actv_layer2 = ReLU(name="idecode_relu2")(image_decoder_norm_layer2)
image_decoder_output = Conv2DTranspose(
    3, (3, 3), padding="same", name="image_output_layer"
)(image_decoder_actv_layer2)


# decoder for text
text_decoder_dense_layer1 = Dense(16, activation="tanh", name="tdecode_dense1")(
    image_text_concat
)
text_reshape = Reshape(target_shape=text_shape_before_concat, name="text_reshape")(
    text_decoder_dense_layer1
)
text_decoder_dense_layer2 = Dense(32, activation="tanh", name="tdecode_dense2")(
    text_reshape
)
text_decoder_output = Dense(13, activation="sigmoid", name="text_output_layer")(
    text_decoder_dense_layer2
)
# decoding_model=Model(inputs=[decoder_input],outputs=[image_decoder_output,text_decoder_output])

ae_sep_output = Model(
    [image_encoder_input, text_encoder_input],
    [image_decoder_output, text_decoder_output],
)

<<<<<<< HEAD
=======
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
import tensorflow as tf

>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b
vectorizer = TfidfVectorizer(stop_words=None)
train_data_copy = train_data
train_data_copy["features"] = train_data_copy.features.apply(lambda x: str(x))
vectors = vectorizer.fit_transform(train_data_copy["features"])
idf = vectorizer.idf_
# build the weight dictionary
new_dict = {}
for c in mlb.classes_:
    if c in vectorizer.vocabulary_.keys():
        new_dict[c] = idf[vectorizer.vocabulary_[c]]
    else:
        new_dict[c] = np.max(idf)
print("\n Printing the TF-IDF for the labels\n\n", new_dict)
weight_freq = {k: v / sum(new_dict.values()) for k, v in new_dict.items()}
print("\nPrinting the weight normalised\n\n")
print(weight_freq)
weight_vector = [v * 1000 for v in new_dict.values()]
tensor_from_list = tf.convert_to_tensor(weight_vector)
tensor_from_list = K.cast(tensor_from_list, "float32")
print("Weight Vector")
print(weight_vector)


def loss_func1(y_true, y_pred):
    # tile sprite loss
    r_loss=K.mean(K.square(y_true - y_pred), axis=[1,2,3])
    loss  =  r_loss
    return loss
    
def loss_func4(y_true,y_pred):
    # multilabel text weighted bce
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    bce_array=-(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))
    weighted_array=bce_array*tensor_from_list
    bce_sum=K.sum(weighted_array,axis=1)
    loss=bce_sum/13.0
    return loss

losses ={'image_output_layer':loss_func1,
          'text_output_layer':loss_func4,
}
#tweak loss weights
lossWeights={'image_output_layer':0.1,
          'text_output_layer':0.9  
        }


def check_nonzero(y_true,y_pred):
    """
    Custom metric
    Returns sum of all embeddings
    """
    return(K.sum(K.cast(y_pred > 0.4, 'int32')))

accuracy={
    'image_output_layer':loss_func1,
    'text_output_layer': check_nonzero
}
from keras import metrics

ae_sep_output.compile(
    optimizer="adam", loss=losses, loss_weights=lossWeights, metrics=accuracy
)

# with loss func 2 that is by using in built cross-entropy loss

es = EarlyStopping(
    monitor="val_text_output_layer_loss", mode="min", verbose=1, patience=2
)

ae_history = ae_sep_output.fit(
    [train_image_batch, train_text_batch],
    [output_image_batch, output_text_batch],
    epochs=10,
    batch_size=25,
    shuffle=True,
    validation_split=0.2,
    callbacks=[es],
)


# build the inference model

decoder_input = Input(shape=(256,))

d_dense = ae_sep_output.get_layer("image_dense")(decoder_input)
d_reshape = ae_sep_output.get_layer("image_reshape")(d_dense)
d_conv1 = ae_sep_output.get_layer("idecode_conv1")(d_reshape)
d_norm1 = ae_sep_output.get_layer("idecode_norm1")(d_conv1)
d_relu1 = ae_sep_output.get_layer("idecode_relu1")(d_norm1)
d_conv2 = ae_sep_output.get_layer("idecode_conv2")(d_relu1)
d_norm2 = ae_sep_output.get_layer("idecode_norm2")(d_conv2)
d_relu2 = ae_sep_output.get_layer("idecode_relu2")(d_norm2)
d_image_output = ae_sep_output.get_layer("image_output_layer")(d_relu2)

t_dense = ae_sep_output.get_layer("tdecode_dense1")(decoder_input)
t_reshape = ae_sep_output.get_layer("text_reshape")(t_dense)
t_dense2 = ae_sep_output.get_layer("tdecode_dense2")(t_reshape)
d_text_output = ae_sep_output.get_layer("text_output_layer")(t_dense2)

decoder_model = Model(inputs=[decoder_input], outputs=[d_image_output, d_text_output])
# decoder_model.summary()

## save model weights, multilabel binarizer

import pickle
# saving
with open("model_tokenizer.pickle", "wb") as handle:
    pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)

## saving the entire architecture model
model_json = ae_sep_output.to_json()
with open("autoencoder_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
ae_sep_output.save_weights("autoencoder_model.h5")
print("Saved Entire Model to disk")

## saving the encoder part
model_json = encoding_model.to_json()
with open("encoder_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoding_model.save_weights("encoder_model.h5")
print("Saved Encoder Model to disk")
## saving the encoder part

model_json = decoder_model.to_json()
with open("decoder_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
decoder_model.save_weights("decoder_model.h5")
print("Saved Decoder Model to disk")

