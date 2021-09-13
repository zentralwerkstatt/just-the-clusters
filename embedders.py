from util import from_device
import numpy as np
import PIL.Image
import torch as t
import torchvision as tv
import torch.nn as nn
import clip


class Embedder_Raw():
    def __init__(self, resolution=32, device="cpu"):
        self.device = device
        self.resolution = resolution
        self.feature_length = self.resolution * self.resolution * 3
        
    def transform(self, img):
        img = img.resize((self.resolution, self.resolution), PIL.Image.ANTIALIAS)
        output = np.array(img).flatten()
        return output.astype(np.uint8).flatten()

class Embedder_VGG19():
    def __init__(self, device="cpu"):
        self.device = device
        self.feature_length = 4096
        self.model = tv.models.vgg19(pretrained=True).to(self.device)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:5])  # VGG19 fc1
        self.model.eval()
        self.transforms = tv.transforms.Compose([tv.transforms.Resize((224, 224)), 
                                                 tv.transforms.ToTensor()])

    def transform(self, img):
        with t.no_grad():
            output = self.model(self.transforms(img).unsqueeze(0).to(self.device))
            return from_device(output).astype(np.float32).flatten()

class Embedder_CLIP():
    def __init__(self, device="cpu"):
        self.device = device
        self.feature_length = 512
        self.model, self.transforms = clip.load("ViT-B/32", device=self.device) # Not using preprocess
        self.image_mean = t.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
        self.image_std = t.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)

    def transform(self, img):
        with t.no_grad():
            input_ = self.transforms(img).unsqueeze(0).to(self.device)
            input_ -= self.image_mean[:, None, None]
            input_ /= self.image_std[:, None, None]
            output = self.model.encode_image(input_)
            output /= output.norm(dim=-1, keepdim=True)
            return from_device(output).astype(np.float32).flatten()