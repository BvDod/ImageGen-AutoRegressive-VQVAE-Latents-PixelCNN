import torch
import torch.nn as nn
import torchvision
from models.vq_vae import VQVAE
import random
import numpy

from models.image_gpt import ImageGPT
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from PIL import Image as im 

import numpy as np

def train_imageGPT(settings_imageGPT, settings_VQVAE):

    device = "cpu"
    print(f"Device: {device}" + "\n")
    """
    # Load models
    model_settings_imageGPT = settings_imageGPT["model_settings"]
    model_settings_imageGPT["num_channels"] =  1
    model_settings_imageGPT["input_shape"] = (32,32)
    model_settings_imageGPT["device"] = device
    model_autoregressive = ImageGPT(model_settings_imageGPT)
    model_autoregressive.load_state_dict(torch.load("model_latest.pt", map_location=torch.device('cpu'), weights_only=True))
    model_autoregressive.to(device)

    model_autoregressive.eval()

    with torch.no_grad():
        image_tokens = torch.Tensor().long()
        padding_tokens = torch.zeros((1,1), dtype=torch.long)
        for i in range((32*32)):

            input_tokens = torch.cat((image_tokens, padding_tokens), dim=1)
            prediction = model_autoregressive(input_tokens, flat_mode=True)
            logits = prediction[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            image_tokens = torch.cat((image_tokens, pred), dim=1)
            if i % 100 == 0:
                print(i)

         
    """
    
    model_settings_VQVAE = settings_VQVAE["model_settings"]
    model_settings_VQVAE["num_channels"] =  3
    model_settings_VQVAE["input_shape"] = (256,256)
    model_settings_VQVAE["device"] = device
    
    model_VQVAE = VQVAE(model_settings_VQVAE)
    model_VQVAE.load_state_dict(torch.load("models/saved_models/celebA/model_best(222).pt", map_location=torch.device('cpu'), weights_only=True))
    model_VQVAE.to(device)
    model_VQVAE.eval()

    with torch.no_grad():
        outputs = model_VQVAE.decode_latents(image_tokens)

    grid = plot_grid_samples_tensor(outputs[:1].cpu())
    
    foldername = "generated_samples/"
    grid = grid.movedim(0,-1)
    image = im.fromarray((grid.cpu().numpy() * 255).astype(np.uint8))
    Path(foldername).mkdir(parents=True, exist_ok=True)
    image.save(f"{foldername}output.png")


def plot_grid_samples_tensor(tensor, grid_size=[8,8]):
    """ Plots a grid of random samples from a tensor with grid size = grid size"""
    grid = torchvision.utils.make_grid(tensor, nrow=grid_size[0])
    return grid




if __name__ == "__main__":
    settings_imageGPT = {
        "dataset": "celebA",
        "save_model": True,
        "model_settings" : {
            "num_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "transformer_layers": 5,
            "attention_heads": 4,
            "vocabulary_size": 512
        }
    }
    settings_VQVAE = {
        "dataset": "celebA",
        "model_settings" : {
            "encoder_architecture": "VIT",
            "decoder_architecture": "VIT",
            "num_hidden": 64,
            "num_residual_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.25,
            "transformer_layers": 5,
            "attention_heads": 4,
            "patch_size": 8,
        }
    }
    train_imageGPT(settings_imageGPT, settings_VQVAE)
