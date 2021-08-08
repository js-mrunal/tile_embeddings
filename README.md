# Tile Embedding: A General Representation for Level Generation.
### Authors: Mrunal Jadhav and Matthew Guzdial 

In  recent  years,  Procedural  Level  Generation  via  Machine Learning (PLGML) techniques have been applied to generate game levels with machine learning. These approaches rely on human-annotated representations of game levels. Creating annotated datasets for games requires domain knowledge and is time-consuming. Hence, though a large number of video games exist, annotated datasets are curated only for a small handful. Thus current PLGML techniques have been explored in limited domains, with Super Mario Bros. as the most common example. To address this problem, we present tile embeddings,  a  unified,  affordance-rich  representation  for  tile-based  2D  games.  To  learn  this  embedding,  we  employ  autoencoders trained on the visual and semantic information oft iles from a set of existing, human-annotated games. We evaluate this representation on its ability to predict affordancesfor unseen tiles, and to serve as a PLGML representation for annotated and unannotated games.

To promote future research and as a contribution to PCGML community, through this repository we provide:
1. An end to end implementation 
2. Level Generation
3. Context data for every unique tile type..
4. Preprocessed Bubble Bobble levels 

<!-- Paper: 
Please cite : -->

### Quick Rundown
1. Data Extraction and Preparation(#data-extraction-and-preparation)
2. Autoencoder Training
3. Level Representation using tile embeddings
4. Bubble Bobble level generation using LSTM 

## Install Dependencies

```
pip install -r requirements.txt
```

## Data Extraction and Preparation
1. Data Extraction and Preparation- The training data for our implementation includes five games: *Super Mario Bros, Kid Icarus, Legend of Zelda, Lode Runner, Megaman*. To train the autoencoder for obtain an embedded representation of tile, we draw on local pixel context and the affordances of the candidate tile. 

a. Local Pixel Context: To extract the 16 * 16 tiles along with its local context, we slide a 48 * 48 window over the level images. The parent dataset for level images is [VGLC](https://github.com/TheVGLC/TheVGLC). However, level images for some games have extra pixels along the vertical/horizontal axis which result in off-centered tile sprite extraction(demonstrated in fig). We perform prelimnary image manipulations on this dataset to fit the dimensions of such level images. Lode Runner levels has 8 * 8 tile size which we upscaled to 16 * 16 using the [PIL](https://pillow.readthedocs.io/en/stable/) library. We provide the preprocessed dataset directory [vglc](https://github.com/js-mrunal/tile_embeddings/tree/main/data/vglc).
<img src="images/data_extraction.png">
b. Affordances: 
We define a single, unified set of 13 tags across the games. The tile character to behaviour mapping is provided as [JSON](https://github.com/js-mrunal/tile_embeddings/tree/main/data/json_files_trimmed_features) files. 
    
Thus the Inputs obtained are as follows: 
<img src="images/inputs.png">

To extract the context for all five games run the following. 
1. Move to directory: context_extraction
```
cd notebooks/context_extraction
```

2. Run the following command in shell
```
python extract_context.py
```

On navigating to the folder *data > context_data >* each game folder should be populated by its visual context sorted by the centre tile.
The json files generated in each game directory is a dictionary with key as the centre tile and value enlisting all possible neighbourhoods it. 

## Training autoencoder

1. The jupyter notebook "notebooks > autoencoder_training.ipynb" provides a step by step guide for autoencoder training. 

2. You can also run the following commands to train the autoencoder and save the weights:

>a. Move to the directory: notebooks
```
cd notebooks/
```
>b. Run the following command in shell
```
python autoencoder_training.py
```

3. Load the directly provided architecture and weights. Sample Notebook: 

## Generating Level Representation using trained autoencoder-Bubble Bobble
The notebook *bubble_bobble_generation.ipynb* demonstrates step by step level generation using tile embedding.
