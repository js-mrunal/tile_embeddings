import os
import glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import json
import pickle
from keras.preprocessing import sequence, image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from utils.data_loading.load_data import get_tile_data

if __name__ == "__main__": 

    data_directory = "../data/context_data/"
    json_directory = "../data/json_files_trimmed_features/"
    data = get_tile_data(data_directory, json_directory)
    print("\nThe size of total data is", data.shape)

    # shuffle the data so that the test data does not contain only one-game 
    data = shuffle(data)
    # split the data into training and testing 
    train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
    print("\nThe size of the train data is ", train_data.shape)
    print("The size of the test data is ", test_data.shape)

    # saving the csv
    data.to_csv("../data/game_data.csv")
    train_data.to_csv("../data/train_data.csv")
    test_data.to_csv("../data/test_data.csv")

