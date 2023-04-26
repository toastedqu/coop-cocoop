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

class CoCoOp(nn.Module):
    def __init__(self, cfg, model, device):
        super().__init__()
        self.n_cls = len(cfg.DATASET.CLASSNAMES)
        self.n_ctx = cfg.TRAIN.N_CTX
        self.dtype = model.dtype
        ctx_dim = model.ln_final.weight.shape[0]
        vis_dim = model.visual.output_dim
        self.device = device
        # img_dim = model.visual.input_resolution
        
        # prompt in CoCoOp with classname at the back looks like: [SOS][V1]...[Vn][CLS][EOS]
        
        # context init (always global in cocoop)
        if cfg.TRAIN.CTX_INIT:
            # fixed init
            ctx_init = cfg.TRAIN.CTX_INIT.replace("_", " ") # The "_" is for fill-in of class name
            with torch.no_grad():
                token_emb = model.token_embedding(clip.tokenize(ctx_init).to(device)).type(model.dtype)
            ctx_vectors = token_emb[0, 1:self.n_ctx+1, :]
            prefix = ctx_init
        else:
            # random init
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=model.dtype)
            nn.init.normal_(ctx_vectors, std=cfg.TRAIN.PARAM_STD)
            prefix = " ".join(["X"]*self.n_ctx)
        
        # context vectors
        self.ctx = nn.Parameter(ctx_vectors)      
          
        # FF (image -> ctx bias)
        self.net = nn.Sequential(
            nn.Linear(vis_dim, vis_dim//16),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim//16, ctx_dim)
        )
        
        # prompt finalization
        classnames = [classname.replace("_", " ") for classname in cfg.DATASET.CLASSNAMES]
        raw_prompts = [prefix + " " + classname + "." for classname in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(raw_prompt).to(device) for raw_prompt in raw_prompts])
        with torch.no_grad():
            token_emb = model.token_embedding(self.tokenized_prompts).type(model.dtype)
        
        # [SOS]
        self.register_buffer("prefix", token_emb[:, :1, :])
        # [CLS][EOS]
        self.register_buffer("suffix", token_emb[:, self.n_ctx+1:, :])
        
    def forward(self, img_feats):
        bias = self.net(img_feats.type(torch.float32)).unsqueeze(1).type(self.dtype) # (batch, 1, dim)
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).to(self.device)                  # (1, n_ctx, dim)
        ctx_shifted = ctx + bias                # (batch, n_ctx, dim)
        
        prompt_embs = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)

            prompt_emb = torch.cat(
                [
                    self.prefix,    # (n_cls, 1, dim)
                    ctx_i,          # (n_cls, n_ctx, dim)
                    self.suffix     # (n_cls, sfx_len, dim)
                ], dim=1
            )
            
            prompt_embs.append(prompt_emb)
        
        return torch.stack(prompt_embs)


class CustomCLIPCoCoOp(nn.Module):
    def __init__(self, cfg, model,device):
        super().__init__()
        self.prompt_learner = CoCoOp(cfg, model,device)
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
        logit_scale = self.model.logit_scale.exp()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        prompt_embs = self.prompt_learner(img_feats)
        
        # normalize
        img_feats = img_feats/img_feats.norm(dim=-1, keepdim=True).type(self.model.dtype)   # It was float16 but model.dtype = float32.
        
        logits = []
        for pts_i, img_i in zip(prompt_embs, img_feats):
            txt_feats = self.encode_text(pts_i, tokenized_prompts)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            l_i = logit_scale * img_i @ txt_feats.t()
            logits.append(l_i)

        return torch.stack(logits)