import numpy as np
import random
random.seed(42)

class RandomAgent:
    def __init__(self, env, seed = None):
        self.env = env
        if seed is not None:
            random.seed(seed)
        
    def act(self, state):
        valid_actions = self.env.get_valid_actions(state)
        return random.choice(valid_actions)
    