import os
import glob
import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from keras.preprocessing import sequence, image
from keras.preprocessing.image import array_to_img, save_img, img_to_array
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

LATENT_DIM=256
BATCH_SIZE=1
KERNEL_SIZE=(3,3)
NUM_FEATURES=13

def get_autoencoder():

    # Defining Encoder
    # image encoder
    image_encoder_input = Input(shape=(48, 48, 3), name="image_input")
    image_encoder_conv_layer1 = Conv2D(
        32, strides=3, kernel_size=KERNEL_SIZE, name="iencode_conv1"
    )(image_encoder_input)
    image_encoder_norm_layer1 = BatchNormalization()(image_encoder_conv_layer1)
    image_encoder_actv_layer1 = ReLU()(image_encoder_norm_layer1)
    image_encoder_conv_layer2 = Conv2D(32, KERNEL_SIZE, padding="same", name="iencode_conv2")(
        image_encoder_actv_layer1
    )
    image_encoder_norm_layer2 = BatchNormalization()(image_encoder_conv_layer2)
    image_encoder_actv_layer2 = ReLU()(image_encoder_norm_layer2)
    image_encoder_conv_layer3 = Conv2D(16, KERNEL_SIZE, padding="same", name="iencode_conv3")(
        image_encoder_actv_layer2
    )
    image_encoder_norm_layer3 = BatchNormalization()(image_encoder_conv_layer3)
    image_encoder_actv_layer3 = ReLU()(image_encoder_norm_layer3)
    image_shape_before_flatten = K.int_shape(image_encoder_actv_layer3)[1:]
    image_flatten = Flatten(name="image_flatten_layer")(image_encoder_actv_layer3)

    # text encoder
    text_encoder_input = Input(shape=(NUM_FEATURES,))
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

    # Defining Decoder
    # decoder for image
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
    text_decoder_output = Dense(NUM_FEATURES, activation="sigmoid", name="text_output_layer")(
        text_decoder_dense_layer2
    )
    # decoding_model=Model(inputs=[decoder_input],outputs=[image_decoder_output,text_decoder_output])

    ae_sep_output = Model(
        [image_encoder_input, text_encoder_input],
        [image_decoder_output, text_decoder_output],
    )

    return encoding_model, ae_sep_output