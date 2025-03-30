import torch
import torch.nn as nn
import math

from models.transformer import TransformerBlock, PositionalEmbedding


class ImageGPT(torch.nn.Module):
    """ 
    Implementation of a vector quantized variational autoencoder following the original paper by A. van den Oord et al.
    """
    def __init__(self, model_settings):
        super(ImageGPT, self).__init__()
        self.model_settings = model_settings
        
        self.n_layers = model_settings["transformer_layers"]
        self.vocabulary_size = model_settings["vocabulary_size"]
        self.embedding_dim = model_settings["embedding_dim"]
        self.heads = model_settings["attention_heads"]
        self.device = self.model_settings["device"]
        self.context_length = self.model_settings["input_shape"][0] * self.model_settings["input_shape"][1]

        self.embedding = nn.Embedding(self.vocabulary_size + 1, self.embedding_dim)
        self.positional_embedding = PositionalEmbedding(self.context_length, model_settings)
        self.transformers = nn.Sequential(*[TransformerBlock(model_settings, causal=True) for i in range(self.n_layers)])

        self.decoder_head = nn.Linear(self.embedding_dim, self.vocabulary_size)

        

    def forward(self, x):
        original_shape = x.shape
       
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) # 2d to 1d
        x = self.embedding(x)
        x = self.positional_embedding(x) + x

        
        sos_token = torch.full((x.shape[0], 1), self.vocabulary_size)
        sos_token = self.embedding(sos_token)
        x = torch.concat([sos_token, x], dim=1)

        x = self.transformers(x)
        x = self.decoder_head(x)[:,1:,]
        x = x.reshape((x.shape[0], original_shape[1], original_shape[2], self.vocabulary_size))
        x = x.movedim(-1, 1)
        return x
        
        