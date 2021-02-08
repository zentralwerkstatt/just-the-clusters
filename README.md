# Just-the-clusters (JTC)

A quick and simple command line utility to create semantically-clustered image plots using pre-trained neural networks like VGG19 and CLIP.

## Installation

- Determine CUDA version suffix (e.g. "+cu110" for 11.0), or leave blank for CPU only (replace below)
- Create Python 3.8 environment (e.g. with `conda create -n "jtc" python=3.8` - Python 3.9 is not supported by numba yet, and thus also not supported by umap-learn)
- Install:
```
pip install torch==1.7.1{version_suffix} torchvision==0.8.2{version_suffix} -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
pip install h5py umap-learn lap
```