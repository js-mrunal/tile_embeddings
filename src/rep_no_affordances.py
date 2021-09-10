import pandas as pd
import numpy as np
import pickle
import glob
import json
import os

from sklearn.utils import shuffle
from keras.preprocessing import sequence, image
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import model_from_json
from keras import backend as K
from collections import Counter
from utils.data_loading.load_data import get_tile_data
from PIL import Image, ImageOps

# load the multilabel binarizer
with open("../model/model_tokenizer.pickle", "rb") as handle:
    mlb = pickle.load(handle)
print("Feature Dictionary Loaded")
total_features = len(mlb.classes_)
print("The feature dictionary has size", total_features)
print("Features", mlb.classes_)

# load entire autoencoder architecture
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

def get_image(path):
    img_without_border = load_img(path)
    img = Image.open(path)
    img_with_border = ImageOps.expand(img_without_border, border=16, fill="black")
    return img_without_border, img_with_border

def level_image_unroll(level_array_padded):
    level_image_unrolled = []
    image_h, image_w, image_c = level_array_padded.shape
    for x in range(0, image_w - 32, 16):
        for y in range(0, image_h - 32, 16):
            context_tile = level_array_padded[y : y + 48, x : x + 48, :]
            level_image_unrolled.append(context_tile)
    return np.array(level_image_unrolled)

def build_game_dataframe(current_game, game_image_dir, image_extension):
    image_ids = set(
        [
            path.split("/")[-1].split(".")[0]
            for path in glob.glob(game_image_dir + "/*" + image_extension)
        ]
    )
    ids = image_ids
    # build a dataframe
    image_paths = [game_image_dir + image_id + ".png" for image_id in ids]
    game_data = pd.DataFrame(columns=["image_path"])
    game_data["image_path"] = image_paths
    assert game_data.shape[0] == len(ids)
    print("\nAll Levels Loaded")
    print("\nTotal Levels for game ", current_game, " detected are ", len(ids))
    return game_data, list(ids)

def generate_unified_rep(current_game,loaded_game_data,game_image_dir, save_dir):
    
    ptr = 0

    idx2embed_map = {}
    idx2tile_map = {}

    for idx in range(len(loaded_game_data)):

        image_path = loaded_game_data.loc[idx]["image_path"]
        level_id = image_path.split("/")[-1].split(".")[0]
        print("\nProcessing level", level_id)
        level_img, level_img_padded = get_image(image_path)
        level_array = img_to_array(level_img)
        level_array_padded = img_to_array(level_img_padded)

        assert level_array.shape[0] % 16 == 0
        assert level_array.shape[1] % 16 == 0
        level_h = level_array.shape[0] / 16
        level_w = level_array.shape[1] / 16
        print("Height ", level_h, "Width ", level_w)
        level_image_expanded = level_image_unroll(level_array_padded)
        print("Expanded level images ", level_image_expanded.shape)

        mapped_text = np.zeros((level_image_expanded.shape[0], 13))
        encoded_level = encoding_model.predict([level_image_expanded, mapped_text])
        print("Encoding dimension", encoded_level.shape)

        for i in range(len(encoded_level)):
            tile_embedding = encoded_level[i]
            tile_sprite = level_image_expanded[i].reshape(48, 48, 3)[
                16 : 16 + 16, 16 : 16 + 16, :
            ]
            idx2embed_map[ptr] = tile_embedding
            idx2tile_map[ptr] = tile_sprite
            ptr += 1

        with open(save_dir + level_id + ".pickle", "wb") as handle:
            pickle.dump(encoded_level, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved ", level_id, " successfully!")

    with open(save_dir + "mappings/idx2embed.pickle", "wb") as handle:
        pickle.dump(idx2embed_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Index to Embedding map saved successfully!")

    with open(save_dir + "mappings/idx2tile.pickle", "wb") as handle:
        pickle.dump(idx2tile_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Index to Tile map saved successfully!")
    print("Extracted unified representation for game ",current_game)

    
if __name__ == "__main__": 
    current_game = "bubble_bobble"
    game_image_dir = "../data/bubble_bobble/"
    save_dir = "../data/unified_rep/bubble_bobble/"
    loaded_game_data, identifiers = build_game_dataframe(
        current_game,
        game_image_dir,
        ".png")

    generate_unified_rep(current_game,loaded_game_data,game_image_dir, save_dir)
    
    print("Saved Bubble Bobble Unified Representation!")
