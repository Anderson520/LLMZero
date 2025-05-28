import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np
from torch.distributions.categorical import Categorical


class TransformerNetwork(nn.Module):
    def __init__(self, 
                 embedding_dim=768,
                 output_dim=4,
                 num_transformer_layers=1,
                 fc_hidden_dims=[512, 256],
                 fc_output_std=0.01
                 ):
        super(TransformerNetwork, self).__init__()
                
        self.embedding_dim = embedding_dim
        
        # Transformer encoder layer (self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=8, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Configurable fully connected layers for Q-values
        fc_layers = []
        input_dim = self.embedding_dim
        for hidden_dim in fc_hidden_dims:
            fc_layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
            fc_layers.append(nn.ReLU())
            input_dim = hidden_dim
        fc_layers.append(layer_init(nn.Linear(input_dim, output_dim), std=fc_output_std))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, embeddings):
        assert isinstance(embeddings, torch.Tensor), f"Expected input type torch.Tensor but got {type(embeddings)}"
        assert embeddings.shape[-1] == self.embedding_dim, f"Expected input shape [batch_size, seq_len, {self.embedding_dim}] but got {embeddings.shape}"
        
        # Pass embeddings through the transformer encoder layer
        transformer_output = self.transformer_encoder(embeddings)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Average the transformer output across the sequence dimension
        pooled_output = transformer_output.mean(dim=1)
        
        # Output Q-values
        q_values = self.fc(pooled_output)  # Shape: [batch_size, num_actions]
        return q_values
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize the weights and biases of a layer.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): Standard deviation for orthogonal initialization.
        bias_const (float): Constant value for bias initialization.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonal initialization
    torch.nn.init.constant_(layer.bias, bias_const)  # Constant bias
    return layer