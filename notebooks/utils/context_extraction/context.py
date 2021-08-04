import pandas as pd
import numpy as np
import glob
import json
import os
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageOps

def extract_context1(current_level,current_img_padded,save_dir,tile_dictionary):

    #for traversing through the text file rows
    x=0
    #for traversing through the image rows
    img_x=0
    imax=len(current_level)
    jmax=len(current_level[0][0])

    #outer loop for the row
    for x in range(imax):
        #for traversing through the text file columns
        y=0
        #image_columns
        img_y=0
        for y in range(jmax):
            #candidate tile character
            current_symbol=current_level[x][0][y]
            #extracting neighbourhood context of candidate tile 
            north=' '
            south=' '
            west=' '
            east=' '
            north_west=' '
            north_east=' '
            south_west=' '
            south_east=' '
            
            ##row 1 of data 
            if x+1<imax and y>0:
                north_west=current_level[x+1][0][y-1]
            if x+1<imax:
                north=current_level[x+1][0][y]
            if x+1<imax and y+1<jmax:
                north_east=current_level[x+1][0][y+1]
            row_1=str(north_west+north+north_east)
            
            #row 2 of data    
            if y>0:
                west=current_level[x][0][y-1]
            if y+1<jmax:
                east=current_level[x][0][y+1]
            row_2=str(west+current_symbol+east)  
            
            ##row 3 of data
            if x>0 and y>0:
                south_west=current_level[x-1][0][y-1]
            if x>0:
                south=current_level[x-1][0][y]        
            if x>0 and y+1<jmax:
                south_east=current_level[x-1][0][y+1]
            row_3=str(south_west+south+south_east)

            #identifier string for the context tile
            sprite_string=str(row_3+row_2+row_1)

            #extract the image 
            tile_context=img_to_array(current_img_padded)[img_x:img_x+48,img_y:img_y+48,:]
            tile_sprite=array_to_img(tile_context)
#             if not tile_context.shape[2]==3:
#                 tile_sprite=tile_sprite.convert('RGB')
            assert img_to_array(tile_sprite).shape == (48,48,3)
            if tile_dictionary.get(current_symbol) is None:
                tile_dictionary[current_symbol]=[]
            if sprite_string not in tile_dictionary.get(current_symbol):
                tile_dictionary[current_symbol].append(sprite_string)
                sprite_dir_path=save_dir+str(current_symbol)+"/"

                if not os.path.exists(sprite_dir_path):
                        os.mkdir(save_dir+str(current_symbol))
                save_img(sprite_dir_path+sprite_string+".png",tile_sprite)
            img_y+=16
        img_x+=16
        
    return tile_dictionary


def extract_context_lr(current_level,current_img_padded,save_dir,tile_dictionary):
    #for traversing through the text file rows
    x=0
    #for traversing through the image rows
    img_x=0
    imax=len(current_level)
    jmax=len(current_level[0][0])

    #outer loop for the row
    for x in range(imax):
        #for traversing through the text file columns
        y=0
        #image_columns
        img_y=0
        for y in range(jmax):
            #current tile_symbol
            current_symbol=current_level[x][0][y]
            #current tile symbol context
            north=' '
            south=' '
            west=' '
            east=' '
            north_west=' '
            north_east=' '
            south_west=' '
            south_east=' '
            if str(current_symbol)=='.':
                current_symbol=str('@')
            ##row 1 of data 
            if x+1<imax and y>0:
                north_west=current_level[x+1][0][y-1]
                if north_west==".":
                    north_west='@'
            if x+1<imax:
                north=current_level[x+1][0][y]
                if north=='.':
                    north='@'
            if x+1<imax and y+1<jmax:
                north_east=current_level[x+1][0][y+1]
                if north_east=='.':
                    north_east='@'
            row_1=str(north_west+north+north_east)
            #row 2 of data    
            if y>0:
                west=current_level[x][0][y-1]
                if west =='.':
                    west="@"
            if y+1<jmax:
                east=current_level[x][0][y+1]
                if east=='.':
                    east='@'
            row_2=str(west+current_symbol+east)  
            ##row 3 of data
            if x>0 and y>0:
                south_west=current_level[x-1][0][y-1]
                if south_west=='.':
                    south_west='@'
            if x>0:
                south=current_level[x-1][0][y] 
                if south=='.':
                    south='@'
            if x>0 and y+1<jmax:
                south_east=current_level[x-1][0][y+1]
                if south_east=='.':
                    south_east="@"
            row_3=str(south_west+south+south_east)
            #identifier string for the context tile
            sprite_string=str(row_3+row_2+row_1)
            #extract the image 
            tile_context=img_to_array(current_img_padded)[img_x:img_x+48,img_y:img_y+48,:]
            tile_sprite=array_to_img(tile_context)
            assert tile_context.shape == (48,48,3)
            if tile_dictionary.get(current_symbol) is None:
                tile_dictionary[current_symbol]=[]
            if sprite_string not in tile_dictionary.get(current_symbol):
                tile_dictionary[current_symbol].append(sprite_string)
                sprite_dir_path=save_dir+str(current_symbol)+"/"
                if not os.path.exists(sprite_dir_path):
                        os.mkdir(save_dir+str(current_symbol))
                save_img(sprite_dir_path+sprite_string+".png",tile_sprite)
            img_y+=16
        img_x+=16
        
    return tile_dictionary
