import pandas as pd
import numpy as np
import glob
import json
import os
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageOps
from utils.context_extraction.context import extract_context1, extract_context_lr

def get_level(path):
    with open(path, "r") as f:
        l = [line.split() for line in f]
    return l

def get_image(path):
    img_without_border = load_img(path)
    img = Image.open(path)
    img_with_border = ImageOps.expand(img_without_border, border=16, fill="black")
    return img_without_border, img_with_border

def update_tile_dictionary(
    game, current_level, current_img_padded, save_dir, tile_dictionary):
    if game == "lode_runner":
        return extract_context_lr(
            current_level, current_img_padded, save_dir, tile_dictionary
        )
    else:
        return extract_context1(
            current_level, current_img_padded, save_dir, tile_dictionary
        )

def save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir):
    tile_dictionary = {}

    for identifier in identifiers:
        image_path = image_dir + identifier + image_extension
        current_img, current_img_padded = get_image(image_path)
        ##figure out the image_dimensions
        data_array = img_to_array(current_img)
        data_array_padded = img_to_array(current_img_padded)
        current_level = get_level(annotation_dir + identifier + ".txt")
        if (
            data_array.shape[0] / len(current_level) == 16.0
            and data_array.shape[1] / len(current_level[0][0]) == 16.0
        ):
            print("Processing", identifier)
            tile_dictionary = update_tile_dictionary(
                game, current_level, current_img_padded, save_dir, tile_dictionary
            )
        else:
            print("Skipping", identifier)
            continue
    # save the tile dictionary
    with open(save_dir + "candidate_tile_context_"+str(game)+".json", "w") as fp:
        json.dump(tile_dictionary, fp, indent=4, sort_keys=True)
    print("Context Data Extracted for game ", game)

<<<<<<< HEAD
    
if __name__ == "__main__":    
    # UNCOMMENT TO EXTRACT DATA
    # SUPER-MARIO-BROS
    game="smb"
    image_extension=".png"
    image_dir="../data/vglc/Super Mario Bros/Original_Fixed/"
    image_paths=glob.glob(image_dir+"*"+image_extension)
    image_identifiers=[x.split("/")[-1].split(".")[0] for x in image_paths]
    annotation_dir="../data/vglc/Super Mario Bros/Processed_Fixed/"
    annotation_paths=glob.glob(annotation_dir+"*.txt")
    annotation_identifiers=[x.split("/")[-1].split(".")[0] for x in annotation_paths]
    identifiers=set(image_identifiers).intersection(set(annotation_identifiers))
    save_dir="../data/context_data/smb/"
    print("Number of levels with annotations detected: ",len(identifiers))
    save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

    # LEGEND OF ZELDA
    game="legend_of_zelda"
    image_extension=".png"
    image_dir="../data/vglc/The Legend of Zelda/Flipped_Data/"
    image_paths=glob.glob(image_dir+"*"+image_extension)
    image_identifiers=[x.split("/")[-1].split(".")[0] for x in image_paths]
    annotation_dir="../data/vglc/The Legend of Zelda/Processed/"
    annotation_paths=glob.glob(annotation_dir+"*.txt")
    annotation_identifiers=[x.split("/")[-1].split(".")[0] for x in annotation_paths]
    identifiers=set(image_identifiers).intersection(set(annotation_identifiers))
    save_dir="../data/context_data/loz/"
    print("Number of levels with annotations detected: ",len(identifiers))
    save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

    # KID ICARUS
    game="kid_icarus"
    image_extension=".bmp"
    image_dir="../data/vglc/Kid Icarus/Original_Fixed/"
    image_paths=glob.glob(image_dir+"*"+image_extension)
    image_identifiers=[x.split("/")[-1].split(".")[0] for x in image_paths]
    annotation_dir="../data/vglc/Kid Icarus/Processed/"
    annotation_paths=glob.glob(annotation_dir+"*.txt")
    annotation_identifiers=[x.split("/")[-1].split(".")[0] for x in annotation_paths]
    identifiers=set(image_identifiers).intersection(set(annotation_identifiers))
    save_dir="../data/context_data/kid_icarus/"
    print("Number of levels with annotations detected: ",len(identifiers))
    save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

    # MEGAMAN
    game = "megaman"
    image_extension = ".bmp"
    image_dir = "../data/vglc/MegaMan/"
    image_paths = glob.glob("../data/vglc/MegaMan/*" + image_extension)
    image_identifiers = [x.split("/")[-1].split(".")[0] for x in image_paths]
    annotation_dir = "../data/vglc/MegaMan/"
    annotation_paths = glob.glob("../data/vglc/MegaMan/*.txt")
    annotation_identifiers = [x.split("/")[-1].split(".")[0] for x in annotation_paths]
    identifiers = set(image_identifiers).intersection(set(annotation_identifiers))
    save_dir = "../data/context_data/megaman/"
    print("Number of levels with annotations detected: ", len(identifiers))
    save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

    # LODE RUNNER
    game="lode_runner"
    image_extension=".png"
    image_dir="../data/vglc/Lode Runner/Original_Fixed/"
    image_paths=glob.glob(image_dir+"*"+image_extension)
    annotation_dir="../data/vglc/Lode Runner/Processed/"
    identifiers=[]
    for image_path in image_paths:
        level_id=image_path.split("/")[-1].split(".")[0]
        identifiers.append(level_id)
    save_dir="../data/context_data/lode_runner/"
    print("Number of levels with annotations detected: ",len(identifiers))
    save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)
=======
# UNCOMMENT TO EXTRACT DATA
# SUPER-MARIO-BROS
game="smb"
image_extension=".png"
image_dir="../data/vglc/Super Mario Bros/Original_Fixed/"
image_paths=glob.glob(image_dir+"*"+image_extension)
image_identifiers=[x.split("/")[-1].split(".")[0] for x in image_paths]
annotation_dir="../data/vglc/Super Mario Bros/Processed_Fixed/"
annotation_paths=glob.glob(annotation_dir+"*.txt")
annotation_identifiers=[x.split("/")[-1].split(".")[0] for x in annotation_paths]
identifiers=set(image_identifiers).intersection(set(annotation_identifiers))
save_dir="../data/context_data/smb/"
print("Number of levels with annotations detected: ",len(identifiers))
save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

# LEGEND OF ZELDA
game="legend_of_zelda"
image_extension=".png"
image_dir="../data/vglc/The Legend of Zelda/Flipped_Data/"
image_paths=glob.glob(image_dir+"*"+image_extension)
image_identifiers=[x.split("/")[-1].split(".")[0] for x in image_paths]
annotation_dir="../data/vglc/The Legend of Zelda/Processed/"
annotation_paths=glob.glob(annotation_dir+"*.txt")
annotation_identifiers=[x.split("/")[-1].split(".")[0] for x in annotation_paths]
identifiers=set(image_identifiers).intersection(set(annotation_identifiers))
save_dir="../data/context_data/loz/"
print("Number of levels with annotations detected: ",len(identifiers))
save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

# KID ICARUS
game="kid_icarus"
image_extension=".bmp"
image_dir="../data/vglc/Kid Icarus/Original_Fixed/"
image_paths=glob.glob(image_dir+"*"+image_extension)
image_identifiers=[x.split("/")[-1].split(".")[0] for x in image_paths]
annotation_dir="../data/vglc/Kid Icarus/Processed/"
annotation_paths=glob.glob(annotation_dir+"*.txt")
annotation_identifiers=[x.split("/")[-1].split(".")[0] for x in annotation_paths]
identifiers=set(image_identifiers).intersection(set(annotation_identifiers))
save_dir="../data/context_data/kid_icarus/"
print("Number of levels with annotations detected: ",len(identifiers))
save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

# MEGAMAN
game = "megaman"
image_extension = ".bmp"
image_dir = "../data/vglc/MegaMan/"
image_paths = glob.glob("../data/vglc/MegaMan/*" + image_extension)
image_identifiers = [x.split("/")[-1].split(".")[0] for x in image_paths]
annotation_dir = "../data/vglc/MegaMan/"
annotation_paths = glob.glob("../data/vglc/MegaMan/*.txt")
annotation_identifiers = [x.split("/")[-1].split(".")[0] for x in annotation_paths]
identifiers = set(image_identifiers).intersection(set(annotation_identifiers))
save_dir = "../data/context_data/megaman/"
print("Number of levels with annotations detected: ", len(identifiers))
save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)

# LODE RUNNER
game="lode_runner"
image_extension=".png"
image_dir="../data/vglc/Lode Runner/Original_Fixed/"
image_paths=glob.glob(image_dir+"*"+image_extension)
annotation_dir="../data/vglc/Lode Runner/Processed/"
identifiers=[]
for image_path in image_paths:
    level_id=image_path.split("/")[-1].split(".")[0]
    identifiers.append(level_id)
save_dir="../data/context_data/lode_runner/"
print("Number of levels with annotations detected: ",len(identifiers))
save_context(game, identifiers, image_dir, image_extension, annotation_dir, save_dir)
>>>>>>> 313595c9947f6e24834176741f5ba4ec9aabc23b

