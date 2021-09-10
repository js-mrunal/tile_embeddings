import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
import json
from keras.preprocessing import sequence, image
from keras.preprocessing.image import array_to_img, save_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import model_from_json
from keras.models import Model
from keras import backend as K
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
from ast import literal_eval
from utils.evaluation_metrics.multilabel.alpha_score import alpha_score
from utils.data_loading.load_data import get_tile_data

def get_pickle_file(path):
    with open(path,"rb") as handle:
        return pickle.load(handle)

if __name__ == "__main__": 

    # set paths
    data_directory = "../data/context_data/"
    json_directory = "../data/json_files_trimmed_features/"
    game_data_path="../data/game_data.csv"
    train_data_path="../data/train_data.csv"
    test_data_path="../data/test_data.csv"


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



    # loading the saved models
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
    # load weights into new model
    decoding_model.load_weights("../model/decoder_model_test.h5")
    print("Loaded Decoder Model from the Disk")



    # inferencing on test data

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

    print("\nMacro Label Based Precision", precision_macro(y_true, y_pred))
    print("Macro Label Based Recall", recall_macro(y_true, y_pred))
    print("Macro Label Based Accuracy", accuracy_macro(y_true, y_pred))

    print("\nMicro Label Based Precision", precision_micro(y_true, y_pred))
    print("Micro Label Based Recall", recall_micro(y_true, y_pred))
    print("Micro Label Based Accuracy", accuracy_micro(y_true, y_pred))

    print("\nExample Based Precision", example_based_precision(y_true, y_pred))
    print("Example Based Recall", example_based_recall(y_true, y_pred))
    print("Example Based Accuracy", example_based_accuracy(y_true, y_pred))