import numpy as np
import PIL.Image
import torch as t
import logging
import sys
import subprocess


# Logging
logging.captureWarnings(True)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

def load_img(path):
    return PIL.Image.open(path).convert("RGB")

def from_device(tensor):
    return tensor.detach().cpu().numpy()

def set_cuda():
    device = "cuda" if t.cuda.is_available() else "cpu"
    return device