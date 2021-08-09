import numpy as np
import pandas as pd
import glob
import os
import json
from sklearn.utils import shuffle
from PIL import Image, ImageOps
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array, array_to_img


def load_features(game_identifier, json_files_path):
    with open(json_files_path + "/" + game_identifier + ".json") as fp:
        sprite_mappings = json.load(fp)
    return sprite_mappings


def get_processed_level(levels_path, level_identifier):
    with open(levels_path + level_identifier + ".txt", "r") as f:
        l = [line.split() for line in f]
    return l


def get_original_level(images_path, level_identifier):
    img_without_border = load_img(images_path + level_identifier + ".png")
    img = Image.open(images_path + level_identifier + ".png")
    img_with_border = ImageOps.expand(img_without_border, border=16, fill="black")
    return img_without_border, img_with_border


def extract_level_context(
    game_identifier, current_level, current_img_padded, sprite_mappings, context_path
):
    level_data = pd.DataFrame(
        columns=[
            "game_identifier",
            "image_path",
            "centre_tile",
            "context_string",
            "features",
        ]
    )
    count = 0
    # for traversing through the text file rows
    x = 0
    # for traversing through the image rows
    img_x = 0
    imax = len(current_level)
    jmax = len(current_level[0][0])
    # creating a dictionary of the context seen
    tile_dictionary = {}
    # outer loop for the row
    for x in range(imax):

        # for traversing through the text file columns
        y = 0
        # image_columns
        img_y = 0
        for y in range(jmax):

            # current tile_symbol
            current_symbol = current_level[x][0][y]

            # current tile symbol context
            north = " "
            south = " "
            west = " "
            east = " "

            north_west = " "
            north_east = " "

            south_west = " "
            south_east = " "

            ##row 1 of data
            if x + 1 < imax and y > 0:
                north_west = current_level[x + 1][0][y - 1]
            if x + 1 < imax:
                north = current_level[x + 1][0][y]
            if x + 1 < imax and y + 1 < jmax:
                north_east = current_level[x + 1][0][y + 1]

            row_1 = str(north_west + north + north_east)

            # row 2 of data
            if y > 0:
                west = current_level[x][0][y - 1]
            if y + 1 < jmax:
                east = current_level[x][0][y + 1]

            row_2 = str(west + current_symbol + east)

            ##row 3 of data
            if x > 0 and y > 0:
                south_west = current_level[x - 1][0][y - 1]
            if x > 0:
                south = current_level[x - 1][0][y]
            if x > 0 and y + 1 < jmax:
                south_east = current_level[x - 1][0][y + 1]

            row_3 = str(south_west + south + south_east)

            # identifier string for the context tile
            sprite_string = str(row_3 + row_2 + row_1)

            # extract the image
            tile_context = img_to_array(current_img_padded)[
                img_x : img_x + 48, img_y : img_y + 48, :
            ]
            tile_sprite = array_to_img(tile_context)

            assert tile_context.shape == (48, 48, 3)

            if tile_dictionary.get(current_symbol) is None:
                tile_dictionary[current_symbol] = []
            tile_dictionary[current_symbol].append(sprite_string)
            sprite_dir_path = context_path + str(current_symbol) + "/"

            if not os.path.exists(sprite_dir_path):
                os.mkdir(context_path + str(current_symbol))

            save_img(sprite_dir_path + sprite_string + ".png", tile_sprite)

            level_data = level_data.append(
                {
                    "game_identifier": game_identifier,
                    "image_path": sprite_dir_path + sprite_string + ".png",
                    "centre_tile": current_symbol,
                    "context_string": sprite_string,
                    "features": sprite_mappings[current_symbol],
                },
                ignore_index=True,
            )
            count += 1
            img_y += 16
        img_x += 16
    return level_data
