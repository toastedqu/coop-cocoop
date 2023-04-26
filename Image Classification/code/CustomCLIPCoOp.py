import clip
import torch
import transformers
import torchvision
import gc
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from tqdm import tqdm
from config import get_cfg_defaults

import numpy as np


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

class CoOp(nn.Module):
    def __init__(self, cfg, model, device,use_context=False):
        super().__init__()
        self.n_cls = len(cfg.DATASET.CLASSNAMES)
        self.n_ctx = cfg.TRAIN.N_CTX
        ctx_dim = model.ln_final.weight.shape[0] #512
        self.device = device
        
        # A prompt in CoOp with classname at the back looks like: [SOS][V1]...[Vn][CLS][EOS]
        # Assume classname is always at the back for now.

        # context init
        if cfg.TRAIN.CTX_INIT:
            # fixed init (assume global ctx)
            ctx_init = cfg.TRAIN.CTX_INIT.replace("_", " ") # The "_" is for fill-in of class name
            with torch.no_grad():
                token_emb = model.token_embedding(clip.tokenize(ctx_init).to(device)).type(model.dtype)
            ctx_vectors = token_emb[0, 1:self.n_ctx+1, :]
            prefix = ctx_init
        else:
            # random init
            if cfg.TRAIN.CSC:
                # class-specific ctx
                ctx_vectors = torch.empty(self.n_cls, self.n_ctx, ctx_dim, dtype=model.dtype)
            else:
                # global ctx
                ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=model.dtype)
            nn.init.normal_(ctx_vectors, std=cfg.TRAIN.PARAM_STD)
            prefix = " ".join(["X"]*self.n_ctx)
        
        # context vectors (THE ONLY PART THAT NEEDS TO BE TRAINED)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # prompt finalization     
        if use_context:
          temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
          classnames = [temp.format(classname.replace("_", " ")) for classname in cfg.DATASET.CLASSNAMES]
          raw_prompts = [prefix + " " + classname + "." for classname in classnames]
        else:
          classnames = [classname.replace("_", " ") for classname in cfg.DATASET.CLASSNAMES]
          raw_prompts = [prefix + " " + classname + "." for classname in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(raw_prompt).to(device) for raw_prompt in raw_prompts])
        # print(self.tokenized_prompts.get_device())
        with torch.no_grad():
            token_emb = model.token_embedding(self.tokenized_prompts).type(model.dtype)
        
        # [SOS]
        self.register_buffer("prefix", token_emb[:, :1, :])
        # [CLS][EOS]
        self.register_buffer("suffix", token_emb[:, self.n_ctx+1:, :])

        
    def forward(self):
        # expand global ctx to match n_cls (i.e., a total of n_cls ctx vectors)
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1).to(self.device)

        prompt_embs = torch.cat(
            [
                self.prefix,    # (n_cls, 1, dim)
                ctx,            # (n_cls, n_ctx, dim)
                self.suffix     # (n_cls, sfx_len, dim)
            ], dim=1
        )
        
        return prompt_embs


class CustomCLIPCoOp(nn.Module):
    def __init__(self, cfg, model,device,use_context=False):
        super().__init__()
        self.prompt_learner = CoOp(cfg, model,device,use_context=False)
        # self.classnames = cfg.DATASET.CLASSNAMES
        self.model = model
        
        # The freezing part was originally done in the training part, but why not just here since we are not modifying anything of CLIP anyway?
        for _,param in self.model.named_parameters():
            param.requires_grad = False
    
    # note that this is nearly identical to the model.encode_text() function from CLIP
    # the only difference is that we already have prompt_embs rather than having to recompute it
    def encode_text(self, prompt_embs, tokenized_prompts):
        x = prompt_embs + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.model.text_projection
        return x
    
    def forward(self, img_feats):
        prompt_embs = self.prompt_learner()

        # encode prompts
        txt_feats = self.encode_text(prompt_embs, self.prompt_learner.tokenized_prompts)
        
        # normalize
        img_feats = img_feats/img_feats.norm(dim=-1, keepdim=True).type(self.model.dtype)   # It was float16 but model.dtype = float32.
        txt_feats = txt_feats/txt_feats.norm(dim=-1, keepdim=True)
        
        logits = self.model.logit_scale.exp() * img_feats @ txt_feats.t()
        return logits