import os
import glob
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.utils import shuffle
from keras.preprocessing import sequence, image
from keras.preprocessing.image import array_to_img, save_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter
from keras import metrics
from utils.data_loading.load_data import get_tile_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from ast import literal_eval
import tensorflow as tf
from keras import metrics
from autoencoder_architecture import get_autoencoder

data_directory = "../data/context_data/"
json_directory = "../data/json_files_trimmed_features/"
EPOCHS=10
BATCH_SIZE=25
LATENT_DIM=256

def loss_func_image(y_true, y_pred):
    # tile sprite loss
    r_loss=K.mean(K.square(y_true - y_pred), axis=[1,2,3])
    loss  =  r_loss
    return loss
    
def loss_func_text(y_true,y_pred):
    # multilabel text weighted bce
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    bce_array=-(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))
    weighted_array=bce_array*tensor_from_list
    bce_sum=K.sum(weighted_array,axis=1)
    loss=bce_sum/13.0
    return loss

def check_nonzero(y_true,y_pred):
    """
    Custom metric
    Returns sum of all embeddings
    """
    return(K.sum(K.cast(y_pred > 0.4, 'int32')))


if __name__ == "__main__": 
    # load the extracted context data
    data = get_tile_data(data_directory, json_directory)
    print("\nThe size of total data is", data.shape)
    data = shuffle(data)
    # split into train-test
    train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
    print("\nThe size of the train data is ", train_data.shape)
    print("The size of the test data is ", test_data.shape)


    # Building feature dictionary consisting of all the features seen across games
    print("Building feature Dictionary..")
    mlb = MultiLabelBinarizer()
    combined_features = np.concatenate(
        [train_data["features"], test_data["features"]], axis=0
    )
    mlb_model = mlb.fit(combined_features)
    num_features = len(mlb_model.classes_)
    print("The feature dictionary has size", num_features)
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
        train_data.shape[0], num_features
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
    test_text_batch = np.array(test_text_batch).reshape(test_data.shape[0], num_features)
    print("\n\nTesting Data Ready")
    print("Train Image batch shape", test_image_batch.shape)
    print("Train Text batch shape", test_text_batch.shape)   

    #initialise the TF-IDF vectorizer to counter imbalanced dataset
    vectorizer = TfidfVectorizer(stop_words=None)
    train_data_copy = train_data
    train_data_copy["features"] = train_data_copy.features.apply(lambda x: str(x))
    vectors = vectorizer.fit_transform(train_data_copy["features"])
    idf = vectorizer.idf_
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

    losses ={'image_output_layer':loss_func_image,
            'text_output_layer':loss_func_text,
    }
    #tweak loss weights
    lossWeights={'image_output_layer':0.1,
            'text_output_layer':0.9  
            }

    accuracy={
        'image_output_layer':loss_func_image,
        'text_output_layer': check_nonzero
    }

    # get the model
    encoding_model, ae_sep_output = get_autoencoder()
    ae_sep_output.compile(
        optimizer="adam", loss=losses, loss_weights=lossWeights, metrics=accuracy
    )
    print("Model Compiled Successfully!")
    es = EarlyStopping(
        monitor="val_text_output_layer_loss", mode="min", verbose=1, patience=2
    )

    ae_history = ae_sep_output.fit(
        [train_image_batch, train_text_batch],
        [output_image_batch, output_text_batch],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        callbacks=[es],
    )

    # build the inference model
    decoder_input = Input(shape=(LATENT_DIM,))

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

    decoding_model = Model(inputs=[decoder_input], outputs=[d_image_output, d_text_output])

    # saving the multi-label binarizer
    with open("../model/model_tokenizer_test.pickle", "wb") as handle:
        pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## saving the entire architecture model
    model_json = ae_sep_output.to_json()
    with open("../model/autoencoder_model_test.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    ae_sep_output.save_weights("../model/autoencoder_model_test.h5")
    print("Saved Entire Model to disk")

    ## saving the encoder part
    model_json = encoding_model.to_json()
    with open("../model/encoder_model_test.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encoding_model.save_weights("../model/encoder_model_test.h5")
    print("Saved Encoder Model to disk")

    ## saving the decoder part
    model_json = decoding_model.to_json()
    with open("../model/decoder_model_test.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    decoding_model.save_weights("../model/decoder_model_test.h5")
    print("Saved Decoder Model to disk")   