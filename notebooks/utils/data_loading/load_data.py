import numpy as np
import pandas as pd
import json
from sklearn.utils import shuffle
import os
import glob


def get_tile_data(game_data_directory, json_directory, shuffle_data=True):

    games = os.listdir(game_data_directory)

    print("Games detected in the parent folder", games)

    tile_data = pd.DataFrame(
        columns=[
            "game_identifier",
            "image_path",
            "centre_tile",
            "context_string",
            "features",
        ]
    )

    for current_game in games:
        print("Current Game", current_game)

        game_context_sprites = os.path.join(game_data_directory, current_game)
        print("Reading mappings")

        with open(json_directory + "/" + current_game + ".json") as fp:
            sprite_mappings = json.load(fp)

        sprite_mappings = sprite_mappings["tiles"]
        print("Json File Loaded")

        print("Reading Sprite Data From", game_context_sprites)

        game_data = {}

        for root, dirs, files in os.walk(game_context_sprites):

            for sprite_directory in dirs:
                game_data[sprite_directory] = glob.glob(
                    game_context_sprites + "/" + sprite_directory + "/*.png"
                )

        #         print("Game Data Keys",game_data)

        for centre_tile in game_data.keys():

            for context in game_data[centre_tile]:
                #                 print("Length of data",len(game_data[centre_tile]))
                context_image_path = context
                #                 print("Context",context_image_path)

                possible_context = context.split("/")[-1].split(".")[0]

                curr_centre = str(context.split("/")[-2])

                #                 print("Centre Tile",curr_centre)
                tile_data = tile_data.append(
                    {
                        "game_identifier": current_game,
                        "image_path": context_image_path,
                        "centre_tile": curr_centre,
                        "context_string": possible_context,
                        "features": sprite_mappings[curr_centre],
                    },
                    ignore_index=True,
                )

    if shuffle_data:
        return shuffle(tile_data)
    return tile_data
