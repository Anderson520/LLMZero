import numpy as np
import random
from utils import elevator_estimate_value

class BFSAgent:
    def __init__(self, env, max_depth=10):
        self.env = env
        self.max_depth = max_depth
        
    def act(self, state):
        valid_actions = self.env.get_valid_actions(state)
        
        values = []
        
        for action in valid_actions:
            checkpoint = self.env.checkpoint()
            # seed = random.randint(0, 1000)
            # self.env.base_env.seed(seed)
            value = self.evaluate(0, action, self.max_depth)
            self.env.restore_checkpoint(checkpoint)
            values.append(value)
            
        best_action = valid_actions[np.argmax(values)]
        
        print(f"Values: {values}")               
        
                
        return best_action
    
    def evaluate(self, value, action, depth):
        next_state, reward, done, _, _ = self.env.step(action)
        
        value += elevator_estimate_value(next_state)
        
        if done or depth == 1:
            return value
        
        valid_actions = self.env.get_valid_actions(next_state)
        
        best_value = -np.inf
        
        for action in valid_actions:
            checkpoint = self.env.checkpoint()
            value = self.evaluate(value, action, depth - 1)
            self.env.restore_checkpoint(checkpoint)
            if value > best_value:
                best_value = value
                
        return reward + best_value
        
        
        
    