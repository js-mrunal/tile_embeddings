import os
import glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import json
import pickle
from ast import literal_eval
from keras.preprocessing import sequence, image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from utils.data_loading.load_data import get_tile_data

game_data_path="../data/game_data.csv"
train_data_path="../data/train_data.csv"
test_data_path="../data/test_data.csv"

if __name__ == "__main__": 
    
    data=pd.read_csv(game_data_path)
    train_data=pd.read_csv(train_data_path)
    test_data=pd.read_csv(test_data_path)
    
    data['features'] = data.features.apply(lambda x: literal_eval(str(x)))
    train_data['features'] = train_data.features.apply(lambda x: literal_eval(str(x)))
    test_data['features'] = test_data.features.apply(lambda x: literal_eval(str(x)))
    
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
    
    print("Train Batches Stored")

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
    
    #save the multilabel binarizer
    with open("../model/model_tokenizer.pickle", "wb") as handle:
        pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nMultilabel binarizer saved")

    # save the input-output training batches
    with open("../data/train_image_batch.pickle", "wb") as handle:
        pickle.dump(train_image_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/train_text_batch.pickle", "wb") as handle:
        pickle.dump(train_text_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/output_image_batch.pickle", "wb") as handle:
        pickle.dump(output_image_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/output_text_batch.pickle", "wb") as handle:
        pickle.dump(output_text_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("\nTrain Batches Stored")

    # save the input-output test batches
    with open("../data/test_image_batch.pickle", "wb") as handle:
        pickle.dump(test_image_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/test_text_batch.pickle", "wb") as handle:
        pickle.dump(test_text_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("\nTest Batches Stored")