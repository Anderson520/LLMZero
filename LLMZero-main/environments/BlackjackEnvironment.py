import gym
import copy

class BlackjackEnvironment(gym.Wrapper):
    '''
    wrapper for Blackjack environment
    '''
    def __init__(self):
        env = gym.make("Blackjack-v1")
        super(BlackjackEnvironment, self).__init__(env)
        
    def reset(self, seed=None):
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        state, reward, done, _, _ = self.env.step(action)
        return state, reward, done, {}
    
    def checkpoint(self):
        # get all relevant properties of the environment
        checkpoint = (
            self.dealer,
            self.player,
            self.dealer_top_card_suit,
            self.dealer_top_card_value_str
        )
        
        return copy.deepcopy(checkpoint)
    
    def restore_checkpoint(self, checkpoint):
        self.dealer, self.player, self.dealer_top_card_suit, self.dealer_top_card_value_str = checkpoint
        
    def get_valid_actions(self, state=None):
        return [0, 1] # 0 for stand, 1 for hit
    
    def get_valid_actions_text(self, state=None):
        return ["stand", "hit"]
    
    def state_to_text(self, state):
        # state is a tuple of (player_sum, dealer_top_card_value, usable_ace)
        player_sum, dealer_top_card_value, usable_ace = state
           
        state_text = f"Player's Hand Total = {player_sum}, "
        state_text += f"Dealer's Upcard = {dealer_top_card_value}, "
        state_text += f"Usable Ace = {usable_ace}"
        
        return state_text
    
    def action_to_text(self, action):
        if action == 0:
            return "stand"
        elif action == 1:
            return "hit"
        else:
            raise ValueError(f"Invalid action: {action}")
        
    def action_txt_to_idx(self, text):
        if text == "stand":
            return 0
        elif text == "hit":
            return 1
        else:
            raise ValueError(f"Invalid action text: {text}")
        
    
    
