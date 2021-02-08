# Just-the-clusters (JTC)

A quick and simple command line utility to create semantically-clustered image plots using pre-trained neural networks like VGG19 and CLIP.

## Installation

- Determine CUDA version suffix (e.g. "+cu110" for 11.0), or leave blank for CPU only
- Create Python 3.8 environment (e.g. with `conda create -n "jtc" python=3.8`)
- Install:
```
pip install torch==1.7.1{version_suffix} torchvision==0.8.2{version_suffix} -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
pip install h5py umap-learn lap
```

## Usage (`-help` output)
```
usage: jtc.py [-h] [--folder FOLDER] [--models MODELS [MODELS ...]] [--thumb_size THUMB_SIZE] [--do_lap]
              [--max_data MAX_DATA]

required arguments:
  --folder FOLDER, -f FOLDER
                        Folder with image files to plot
  --models MODELS [MODELS ...], -m MODELS [MODELS ...]
                        Model(s) to extract embeddings, chose from: clip, vgg19, raw (list)

optional arguments:
  --thumb_size THUMB_SIZE
                        Max. size of thumbnail
  --do_lap              Arrange thumbnails in grid (via Jonker-Volgenant algorithm)
  --max_data MAX_DATA, -n MAX_DATA
                        Only plot n random images
```