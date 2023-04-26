import clip
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from tqdm import tqdm
from config import get_cfg_defaults

CUSTOM_TEMPLATES = {
    'oxfordpet': 'a photo of a {}, a type of pet.',
    'flower102': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'dtd': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, clip_model,device):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.device = device
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.cfg.DATASET.CLASSNAMES]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIPAdapter(nn.Module):

    def __init__(self, cfg, clip_model,device):
        super().__init__()
        # self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model,device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(1024, 4).to(clip_model.dtype)

        # for name, param in self.image_encoder.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad_(False)

        for name, param in self.text_encoder.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
            
    def forward(self, image_features,ratio = 0.2):
        # image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)
        
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits