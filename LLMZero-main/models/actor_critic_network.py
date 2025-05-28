import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np
from torch.distributions.categorical import Categorical
from models.transformer_network import TransformerNetwork

class NLActorCriticNetwork(nn.Module):
    def __init__(self, 
                 encoder_name='sentence-transformers/all-mpnet-base-v2',
                 action_space_n=4,
                 num_transformer_layers_shared=0,
                 num_transformer_layers_actor=1,
                 num_transformer_layers_critic=1,
                 fc_hidden_dims=[512, 256],
                 device='cuda'
                 ):
        super(NLActorCriticNetwork, self).__init__()
                
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encode_model = AutoModel.from_pretrained(encoder_name).to(device).eval()
        self.device = device   
        self.embedding_dim = self.encode_model.config.hidden_size
        
        transformer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=8, dim_feedforward=2048
        )
        
        self.shared_encoder = \
            nn.TransformerEncoder(transformer, num_layers=num_transformer_layers_shared).to(device) \
            if num_transformer_layers_shared > 0 else nn.Identity()
        
        self.actor = TransformerNetwork(
            embedding_dim=self.embedding_dim,
            output_dim=action_space_n,
            num_transformer_layers=num_transformer_layers_actor,
            fc_hidden_dims=fc_hidden_dims
        ).to(device)
        
        self.critic = TransformerNetwork(           
            embedding_dim=self.embedding_dim,
            output_dim=1,
            num_transformer_layers=num_transformer_layers_critic,
            fc_hidden_dims=fc_hidden_dims,
            fc_output_std=1.0
        ).to(device)
            
    def get_value(self, states: List[str]):
        """Calculate the estimated value for a given state.

        Args:
            states (List[str]): List of states as strings, shape: [batch_size]

        Returns:
            torch.Tensor: Estimated value for the state, shape: (batch_size, 1)
        """
        shared_embeddings = self.get_shared_embeddings(states)
        
        return self._get_value(shared_embeddings)
    
    def _get_value(self, shared_embeddings: torch.Tensor):
        """ get value from shared embeddings """
        return self.critic(shared_embeddings)
    
    def get_probs(self, states: List[str]):
        """Calculate the action probabilities for a given state.

        Args:
            states (List[str]): List of states as strings, shape: [batch_size]

        Returns:
            torch.distributions.Categorical: Categorical distribution over actions
        """
        shared_embeddings = self.get_shared_embeddings(states)
        
        return self._get_probs(shared_embeddings)
    
    def _get_probs(self, shared_embeddings: torch.Tensor):
        """ get action probabilities from shared embeddings """
        return Categorical(logits=self.actor(shared_embeddings))
        
    def tokenize_states(self, states: List[str]):
        """ Tokenize list of states (str) and return as dictionary of tensors 
        Args:
            states (List[str]): List of states as strings, shape: [batch_size]
            
        Returns:
            states_tokenized (Dict[str, torch.Tensor]): Dictionary of tokenized states as tensors
        """
                
        states_tokenized = self.tokenizer(list(states), padding=True, truncation=True, return_tensors="pt")
        states_tokenized = {k: v.to(self.device) for k, v in states_tokenized.items()}
        
        return states_tokenized
    
    def get_shared_embeddings(self, states: List[str]):
        """ get shared embeddings """
        states_tokenized = self.tokenize_states(states)
        
        with torch.no_grad():
            state_embeddings = self.encode_model(**states_tokenized).last_hidden_state
            
        shared_embeddings = self.shared_encoder(state_embeddings)
        
        return shared_embeddings
    
    def get_action(self, probs: Categorical):
        """Sample an action from the action distribution.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Sampled action, shape: (batch_size, 1)
        """
        sample = probs.sample()    # shape: [batch_size]
        return sample
    
    def get_action_logprob(self, probs: Categorical, actions: torch.Tensor):
        """Compute the log probability of a given action.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.
            actions (torch.Tensor): Selected action, shape: (batch_size, 1)

        Returns:
            torch.Tensor: Log probability of the action, shape: (batch_size, 1)
        """
        return probs.log_prob(actions)
    
    def get_entropy(self, probs: Categorical):
        """Calculate the entropy of the action distribution.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Entropy of the distribution, shape: (batch_size, 1)
        """
        return probs.entropy()
    
    def get_action_logprob_entropy(self, states: List[str]):
        """Get action, log probability, and entropy for a given state.

        Args:
            states (List[str]): List of states as strings, shape: [batch_size]

        Returns:
            tuple: (action, logprob, entropy)
                - action (torch.Tensor): Sampled action.
                - logprob (torch.Tensor): Log probability of the action.
                - entropy (torch.Tensor): Entropy of the action distribution.
        """
        
        probs = self.get_probs(states)  # Get action probabilities
        action = self.get_action(probs)  # Sample an action
        logprob = self.get_action_logprob(probs, action)  # Compute log probability of the action
        entropy = self.get_entropy(probs)  # Compute entropy of the action distribution
        return action, logprob, entropy  # Return action, log probability, and entropy