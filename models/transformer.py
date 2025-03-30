import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PatchEmbedding(nn.Module):
    def __init__(self, model_settings):
        """ 
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = (model_settings["patch_size"], model_settings["patch_size"])
        self.num_channels = model_settings["num_channels"]
        self.embedding_size = model_settings["num_hidden"]
        
        self.num_patches = np.prod(model_settings["input_shape"]) // np.prod(self.patch_size)

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size[0])
        self.linear1 = nn.Linear(np.prod(self.patch_size) * self.num_channels, self.embedding_size)

    def forward(self, x):
        #TODO: test with channels > 1
        x = self.unfold(x)          # Extract patches
        x = torch.movedim(x, 1,-1)  # Move patch pixels to last dim
        x = self.linear1(x)         # Create embeddings from patches
        
    
        return x
    
class PositionalEmbedding(nn.Module):
    def __init__(self, patch_embeddings, model_settings):

        super(PositionalEmbedding, self).__init__()
        self.num_embeddings = patch_embeddings 
        self.embedding_dim = model_settings["embedding_dim"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, device=self.device)

    def forward(self, x):
        positional_ints = torch.arange(0, self.num_embeddings, requires_grad=False, device=self.device
                                       ).repeat(x.shape[0], 1)
        embedding = self.embedding(positional_ints)
        return embedding


class TransformerBlock(nn.Module):
    def __init__(self, model_settings, dropout=0.1, causal=False):
        """ 
        Classic Transformer Block, optional argument to use causal version of transformer
        """
        super(TransformerBlock, self).__init__()
        self.embedding_dim = model_settings["embedding_dim"]
        self.heads = model_settings["attention_heads"]
        hidden_dim = self.embedding_dim * 4
        self.causal = causal
        self.device = model_settings["device"]
        self.context_length = model_settings["context_length"]

        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim)
        self.attention = nn.MultiheadAttention(self.embedding_dim, self.heads,
                                          dropout=dropout, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim)
        
        if self.causal:
            seq_len = self.context_length + 1 # for sos
            attention_mask = torch.full((seq_len,seq_len), -float("Inf"), device=self.device)
            self.attention_mask = torch.triu(attention_mask, diagonal=1)
        

        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.Dropout(dropout))

    def forward(self, x_in):
        x = self.layer_norm_1(x_in)
       
        if not self.causal:
            x = x_in + self.attention(x, x, x)[0]
        elif self.causal:
            x = x_in + self.attention(x, x, x, attn_mask=self.attention_mask)[0]

        x = x + self.linear(self.layer_norm_2(x))

        return x   