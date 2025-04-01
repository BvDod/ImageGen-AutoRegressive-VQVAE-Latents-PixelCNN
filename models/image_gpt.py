import torch
import torch.nn as nn
import math

from models.transformer import TransformerBlock, PositionalEmbedding


class ImageGPT(torch.nn.Module):
    """ 
    Implementation of ImageGPT-like model to autoregressively predict next pixels (discrete embeddings in our case)
    """
    def __init__(self, model_settings):
        super(ImageGPT, self).__init__()
        self.model_settings = model_settings
        
        # Settings
        self.n_layers = model_settings["transformer_layers"]
        self.vocabulary_size = model_settings["vocabulary_size"]
        self.embedding_dim = model_settings["embedding_dim"]
        self.heads = model_settings["attention_heads"]
        self.device = self.model_settings["device"]
        self.context_length = self.model_settings["input_shape"][0] * self.model_settings["input_shape"][1]
        model_settings["context_length"] = self.context_length

        # Layers
        self.embedding = nn.Embedding(self.vocabulary_size + 1, self.embedding_dim)
        self.positional_embedding = PositionalEmbedding(self.context_length, model_settings)
        self.transformers = nn.Sequential(*[TransformerBlock(model_settings, causal=True) for i in range(self.n_layers)])
        self.decoder_head = nn.Linear(self.embedding_dim, self.vocabulary_size)
    
    def get_sos_token_embeddings(self, shape):
        sos_tokens = torch.full(shape, self.vocabulary_size)
        sos_tokens = self.embedding(sos_tokens)
        return sos_tokens


    def forward(self, x, flat_mode = False):
        original_shape = x.shape
        
        if not flat_mode:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) # 2d to 1d
        
        x = self.embedding(x)

        # Adding sos token to start of each images
        
        sos_embeddings = self.get_sos_token_embeddings((x.shape[0], 1))
        x = torch.concat([sos_embeddings, x[:,:-1,:]], dim=1)
        # We removed the last token to effectively shift the whole image by one pixel.'
        # This achieves: 1. to be prediced pixel cant use own pixel value as input, 2. last pixel is never needed as input

        x = self.positional_embedding(x) + x
        
        x = self.transformers(x)
        x = self.decoder_head(x)
        if not flat_mode:
            x = x.reshape((x.shape[0], original_shape[1], original_shape[2], self.vocabulary_size)) # 1d to 2d
            x = x.movedim(-1, 1)
        return x
        
        