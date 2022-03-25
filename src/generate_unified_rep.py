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

def get_level(path):
    with open(path, "r") as f:
        l = [line.split() for line in f]
    return l

def get_image(path):
    img_without_border = load_img(path)
    img = Image.open(path)
    img_with_border = ImageOps.expand(img_without_border, border=16, fill="black")
    return img_without_border, img_with_border

def get_unified_representation(current_level, current_img_padded, feature_mapping):
    en_height = len(current_level)
    en_width = len(current_level[0][0])
    level_encoding = np.zeros((en_height, en_width, 256))
    count = 0
    # for traversing through the text file rows
    x = 0
    # for traversing through the image rows
    img_x = 0
    imax = len(current_level)
    jmax = len(current_level[0][0])
    # outer loop for the row
    for x in range(imax):
        # for traversing through the text file columns
        y = 0
        # image_columns
        img_y = 0
        for y in range(jmax):
            # current tile_symbol
            current_symbol = current_level[x][0][y]
            # extract the image
            tile_context = img_to_array(current_img_padded)[
                img_x : img_x + 48, img_y : img_y + 48, :
            ]

            tile_context = np.array(tile_context)
            tile_context = np.expand_dims(tile_context, axis=0)
            tile_feature = mlb.transform([feature_mapping[current_symbol]])
            tile_feature = np.array(tile_feature)
            assert tile_context.shape == (1, 48, 48, 3)
            assert tile_feature.shape == (1, 13)
            embedding = encoding_model.predict([tile_context, tile_feature])
            level_encoding[x][y] = embedding

            count += 1
            img_y += 16
        print("x,y", x, y)
        img_x += 16
    return level_encoding

def build_game_dataframe(
    current_game, game_image_dir, representation_dir, json_data_dir, img_extension
):

    image_ids = set(
        [
            path.split("/")[-1].split(".")[0]
            for path in glob.glob(game_image_dir + "/*" + img_extension)
        ]
    )
    rep_ids = set(
        [
            path.split("/")[-1].split(".")[0]
            for path in glob.glob(representation_dir + "/*.txt")
        ]
    )
    ids = image_ids.intersection(rep_ids)
    print(ids)

    # build a dataframe
    game_data = pd.DataFrame(columns=["image_path", "representation_path"])

    for level_id in ids:
        game_data = game_data.append(
            {
                "image_path": game_image_dir + "/" + level_id + img_extension,
                "representation_path": representation_dir + "/" + level_id + ".txt",
            },
            ignore_index=True,
        )

    assert game_data.shape[0] == len(ids)

    print("\nAll Levels Loaded")
    print("\nTotal Levels for game ", current_game, " detected are ", len(ids))

    # get sprite mapping
    with open(json_data_dir + "/" + current_game + ".json") as fp:
        sprite_mappings = json.load(fp)

    sprite_mappings = sprite_mappings["tiles"]
    print("\nJson File Loaded")

    return game_data, sprite_mappings

def generate_unified_rep(loaded_game_data, sprite_mappings, save_dir):

    for ptr in range(loaded_game_data.shape[0]):
    # for ptr in range(1):

        image_path = loaded_game_data["image_path"].iloc[ptr]
        representation_path = loaded_game_data["representation_path"].iloc[ptr]
        level_id = image_path.split("/")[-1].split(".")[0]
        print(level_id)

        current_img, current_img_padded = get_image(image_path)

        ##figure out the image_dimensions
        data_array = img_to_array(current_img)
        data_array_padded = img_to_array(current_img_padded)

        current_level = get_level(representation_path)

        assert data_array.shape[0] / (len(current_level)) == 16
        assert data_array.shape[1] / len(current_level[0][0]) == 16
        print("Getting Unified Representation for ", level_id)
        level_unified_rep = get_unified_representation(
            current_level, current_img_padded, sprite_mappings
        )

        with open(save_dir + level_id + ".pickle", "wb") as handle:
            pickle.dump(level_unified_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Saved", level_id)

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

current_game = "loz"
game_image_dir = "../data/vglc/The Legend of Zelda/Flipped_Data"
representation_dir = "../data/vglc/The Legend of Zelda/Processed"
json_data_directory = "../data/json_files_trimmed_features/"
img_extension = ".png"
save_dir = "../data/unified_rep/loz/"
loaded_game_data, sprite_mappings = build_game_dataframe(
    current_game, game_image_dir, representation_dir, json_data_directory, img_extension
)
generate_unified_rep(loaded_game_data, sprite_mappings, save_dir)

current_game = "kid_icarus"
game_image_dir = "../data/vglc/Kid Icarus/Original_Fixed"
representation_dir = "../data/vglc/Kid Icarus/Processed"
json_data_directory = "../data/json_files_trimmed_features/"
img_extension = ".bmp"
save_dir = "../data/unified_rep/kid_icarus/"
loaded_game_data, sprite_mappings = build_game_dataframe(
    current_game,
    game_image_dir,
    representation_dir,
    json_data_directory,
    img_extension,
)
generate_unified_rep(loaded_game_data, sprite_mappings, save_dir)

current_game = "smb"
game_image_dir = "../data/vglc/Super Mario Bros/Original_Fixed"
representation_dir = "../data/vglc/Super Mario Bros/Processed_Fixed"
json_data_directory = "../data/json_files_trimmed_features/"
img_extension = ".png"
save_dir = "../data/unified_rep/smb/"
loaded_game_data, sprite_mappings = build_game_dataframe(
    current_game,
    game_image_dir,
    representation_dir,
    json_data_directory,
    img_extension,
)
generate_unified_rep(loaded_game_data, sprite_mappings, save_dir)

current_game = "megaman"
game_image_dir = "../data/vglc/MegaMan"
representation_dir = "../data/vglc/MegaMan"
json_data_directory = "../data/json_files_trimmed_features/"
img_extension = ".bmp"
save_dir = "../data/unified_rep/megaman/"
loaded_game_data, sprite_mappings = build_game_dataframe(
    current_game,
    game_image_dir,
    representation_dir,
    json_data_directory,
    img_extension,
)
generate_unified_rep(loaded_game_data, sprite_mappings, save_dir)

current_game = "lode_runner"
game_image_dir = "../data/vglc/Lode Runner/Original_Fixed"
representation_dir = "../data/vglc/Lode Runner/Processed"
json_data_directory = "../data/json_files_trimmed_features/"
img_extension = ".png"
# save_dir = "../data/unified_rep/megaman/"
save_dir = "../data/unified_rep/lode_runner/"
loaded_game_data, sprite_mappings = build_game_dataframe(
    current_game,
    game_image_dir,
    representation_dir,
    json_data_directory,
    img_extension,
)
generate_unified_rep(loaded_game_data, sprite_mappings, save_dir)
