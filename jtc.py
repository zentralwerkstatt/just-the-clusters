# Determine cuda version suffix (e.g. "+cu110" for 11.0), or leave blank for CPU only
# conda create -n "jtc" python=3.8
# pip install torch==1.7.1{torch_version_suffix} torchvision==0.8.2{torch_version_suffix} -f https://download.pytorch.org/whl/torch_stable.html
# (1.7.1 is now stable version, so should work without versions specified)
# pip install git+https://github.com/openai/CLIP.git
# pip install h5py umap-learn lap

from embedders import Embedder_VGG19, Embedder_CLIP, Embedder_Raw
from util import set_cuda, load_img, log

import os
import PIL
import numpy as np
from math import floor, sqrt
import random
import argparse

import tqdm
import h5py
import umap
import lap
from scipy.spatial.distance import cdist

device = set_cuda()
log.info(f"Device: {device}")

parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument("--folder", "-f", help="Folder with image files to plot")
required.add_argument("--models", "-m", help="Model(s) to extract embeddings, chose from: clip, vgg19, raw (list)", nargs='+')
optional.add_argument("--thumb_size", help="Max. size of thumbnail", type=int, default=64)
optional.add_argument("--do_lap", help="Arrange thumbnails in grid (via Jonker-Volgenant algorithm)", action='store_true')
optional.add_argument("--max_data", "-n", help="Only plot n random images", type=int, default=0)
args = parser.parse_args()

folder = args.folder
thumb_size = args.thumb_size
max_data = args.max_data
do_lap = args.do_lap

log.info("Loading models (might download on first run)")
embedders = {}
if "clip" in args.models: embedders["clip"] = Embedder_CLIP(device=device)
if "vgg19" in args.models: embedders["vgg19"] = Embedder_VGG19(device=device)
if "raw" in args.models: embedders["raw"] = Embedder_Raw(resolution=32, device=device)

log.info("Getting file paths and checking files")
valid_files = []
for root, dirs, files in os.walk(folder):
    for file_ in files:
        try:
            img = PIL.Image.open(f"{root}/{file_}")
            valid_files.append(f"{root}/{file_}")
        except:
            continue

if max_data: 
    log.info("Arranging data")
    random.shuffle(valid_files)
    valid_files = valid_files[:max_data]

log.info("Allocating space")
embs = h5py.File("cache.h5py", "w")
for emb_type, embedder in embedders.items():
    data = np.zeros((len(valid_files), embedder.feature_length))
    embs.create_dataset(emb_type, compression="lzf", data=data)

log.info("Extracting features")
for i, file_ in enumerate(tqdm.tqdm(valid_files)):
    img = load_img(file_)
    for emb_type, embedder in embedders.items():
        embs[emb_type][i] = embedder.transform(img)

log.info("Reducing dimensionality (this can take a while)")
reduced_embs = {}
for emb_type in embedders:
    reducer = umap.UMAP()
    reduced_embs[emb_type] = reducer.fit_transform(embs[emb_type])
    # Normalize to 0,1
    reduced_embs[emb_type] -= reduced_embs[emb_type].min(axis=0)
    reduced_embs[emb_type] /= reduced_embs[emb_type].max(axis=0)

log.info("Plotting images")
for emb_type in embedders:
    features = reduced_embs[emb_type]

    if do_lap:
        # https://gist.github.com/vmarkovtsev/74e3a973b19113047fdb6b252d741b42
        # https://github.com/gatagat/lap
        
        grid_size = floor(sqrt(features.shape[0]))
        
        # Cut excess data points
        samples = grid_size*grid_size
        valid_files = valid_files[:samples]
        features = features[:samples]
        
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))).reshape(-1, 2)
        cost_matrix = cdist(grid, features, "sqeuclidean").astype(np.float32)
        cost, row_asses, col_asses = lap.lapjv(cost_matrix)
        features = grid[col_asses]

    # Determine max possible size (as if thumbs arranged without overlap)
    max_size = thumb_size * floor(sqrt(len(features)))
    max_emb_value = abs(np.max(features))
    min_emb_value = abs(np.min(features))
    
    # Calculate size of the plot (images are anchored at upper left corner)
    canvas_size = int((max_emb_value + min_emb_value) * max_size) + thumb_size

    # Create canvas and plot images
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    for i, file_ in enumerate(tqdm.tqdm(valid_files)):
        img = load_img(file_)
        img.thumbnail((thumb_size, thumb_size))
        npimg = np.array(img)
        y = int((features[i,0] + min_emb_value) * max_size)
        x = int((features[i,1] + min_emb_value) * max_size)   
        canvas[y:y+npimg.shape[0],x:x+npimg.shape[1],:] = npimg

    PIL.Image.fromarray(canvas).save(f"{emb_type}.jpg")