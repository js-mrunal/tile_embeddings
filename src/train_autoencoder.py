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
import pickle
from keras.layers import ReLU, Reshape, Conv2DTranspose, Concatenate, Multiply
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics
from collections import Counter
from keras import metrics
from utils.data_loading.load_data import get_tile_data
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
import tensorflow as tf

data_directory = "../data/context_data/"
json_directory = "../data/json_files_trimmed_features/"
game_data_path="../data/game_data.csv"
train_data_path="../data/train_data.csv"
test_data_path="../data/test_data.csv"

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

def get_pickle_file(path):
    with open(path,"rb") as handle:
        return pickle.load(handle)
    
if __name__ == "__main__":

    # data loading
    data=pd.read_csv(game_data_path)
    train_data=pd.read_csv(train_data_path)
    test_data=pd.read_csv(test_data_path)
    data['features'] = data.features.apply(lambda x: literal_eval(str(x)))
    train_data['features'] = train_data.features.apply(lambda x: literal_eval(str(x)))
    test_data['features'] = test_data.features.apply(lambda x: literal_eval(str(x)))

    # loading multi-label binarizer
    mlb=get_pickle_file("../model/model_tokenizer.pickle")
    print("Feature Dictionary Loaded")
    total_features = len(mlb.classes_)
    print("The feature dictionary has size", total_features)
    print("Features", mlb.classes_)

    # loading the batches
    # training 
    train_image_batch=get_pickle_file("../data/train_image_batch.pickle")
    train_text_batch=get_pickle_file("../data/train_text_batch.pickle")
    output_image_batch=get_pickle_file("../data/output_image_batch.pickle")
    output_text_batch=get_pickle_file("../data/output_text_batch.pickle")

    #testing
    test_image_batch=get_pickle_file("../data/test_image_batch.pickle")
    test_text_batch=get_pickle_file("../data/test_text_batch.pickle")

    print("\Training Testing Batches loaded")
    print("Train Image batch shape", train_image_batch.shape)
    print("Train Text batch shape", train_text_batch.shape)
    print("Train Output Image batch shape", output_image_batch.shape)
    print("Train Output Text batch shape", output_text_batch.shape)
    print("Test Image batch shape", test_image_batch.shape)
    print("Test Text batch shape", test_text_batch.shape)   

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

    ae_sep_output.compile(
        optimizer="adam", loss=losses, loss_weights=lossWeights, metrics=accuracy
    )

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
    
    # saving the entire architecture model
    model_json = ae_sep_output.to_json()
    with open("../model/autoencoder_model_test.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    ae_sep_output.save_weights("../model/autoencoder_model_test.h5")
    print("Saved Entire Model to disk")

    # saving the encoder part
    model_json = encoding_model.to_json()
    with open("../model/encoder_model_test.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encoding_model.save_weights("../model/encoder_model_test.h5")
    print("Saved Encoder Model to disk")
    # saving the encoder part

    model_json = decoder_model.to_json()
    with open("../model/decoder_model_test.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    decoder_model.save_weights("../model/decoder_model_test.h5")
    print("Saved Decoder Model to disk")