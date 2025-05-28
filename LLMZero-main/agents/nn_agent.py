import torch
import torch.nn as nn
import numpy as np

from models.transformer_network import TransformerNetwork
from models.actor_critic_network import NLActorCriticNetwork

class NNAgent:
    '''
    Neural Network Agent
    '''
    def __init__(self, env, cfg = None):
        self.cfg = {
            "encoder_name": 'sentence-transformers/all-mpnet-base-v2',
            "network_type": "actor_critic",
            "model_path": "weights/elevator_ppo_Iteration_1552.pt",
            "network_params": {
                "num_transformer_layers_shared": 0,
                "num_transformer_layers_actor": 1,
                "num_transformer_layers_critic": 1,
                "fc_hidden_dims": [512, 256],
                "action_space_n": 4,
                "device": 'cuda',
            }
        }
        
        self.env = env
        
        if cfg is not None:
            self.cfg.update(cfg)
        
        if self.cfg["network_type"] == "actor_critic":
            self.network = NLActorCriticNetwork(**self.cfg["network_params"])
            state_dict = torch.load(self.cfg["model_path"])
            self.network.load_state_dict(state_dict, strict=False)            
        else:
            raise ValueError(f"Invalid network type: {self.cfg['network_type']}")
        
    def act(self, state):
        state_text = self.env.state_to_text(state)
        
        action = self.network.get_probs([state_text]).probs[0]
        
        #select greedy action
        action = np.argmax(action.cpu().detach().numpy())
        
        return action
            
    
            
        
        