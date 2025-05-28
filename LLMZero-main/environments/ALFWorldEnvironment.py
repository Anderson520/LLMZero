"""Adapted from https://github.com/alfworld/alfworld"""

import gym
import alfworld.agents.environment as environment
import yaml
import copy

from alfworld.agents.modules.generic import EpisodicCountingMemory, ObjCentricEpisodicMemory

class ALFWorldEnvironment(gym.Wrapper):
    '''
    wrapper for ALFWorld environment
    '''
    def __init__(self, config_path='configs/alfworld_env.yaml', overwrite_env=None):
        self.config = self._load_config(config_path)
        env = self._init_alfworld_env(self.config, overwrite_env)

        self.env_stack = []
        self.taskdir = ''

        self.state_history = []
        self.action_history = []
        self.episodic_counting_memory = EpisodicCountingMemory() # Tracks newly explored states
        self.obj_centric_episodic_counting_memory = ObjCentricEpisodicMemory() # Tracks newly explored objects

        super().__init__(env)
    
    def _init_alfworld_env(self, config, overwrite_env=None):
        env_type = config['env']['type']
        if overwrite_env is not None:
            config['dataset']['data_path'] = overwrite_env
        env = getattr(environment, env_type)(config, train_eval='train')
        env = env.init_env(batch_size=1) # batch_size = how many environments to run in parallel
        return env

    def reset(self):
        self.state_history = []
        self.action_history = []
        self.episodic_counting_memory.reset()
        self.obj_centric_episodic_counting_memory.reset()

        state, infos = self.env.reset()
        self.taskdir = self._extract_taskdir(infos)

        # Add initial state to memory
        self.episodic_counting_memory.push(state)
        self.obj_centric_episodic_counting_memory.push(state)

        state = self._map_state(state, infos)
        self.state_history.append(state)
        return state, infos
    
    def _extract_taskdir(self, infos):
        gamefile = infos['extra.gamefile'][0]
        taskdir = '/'.join(gamefile.split('/')[:-1])
        return taskdir

    def checkpoint(self):
        '''
        return a checkpoint of the current environment state
        '''
        # Store current environment
        self.env_stack.append(self.env)

        # Initialize environment with only current task
        checkpoint_config = copy.deepcopy(self.config)
        checkpoint_config['dataset']['data_path'] = self.taskdir
        self.env = self._init_alfworld_env(checkpoint_config)

        # Take steps to reach state of current environment
        self.env.reset()
        for action in self.action_history:
            self.env.step(action)

        # Return action_history, episodic_counting_memory, obj_centric_episodic_counting_memory
        return (copy.deepcopy(self.state_history),
                copy.deepcopy(self.action_history),
                copy.deepcopy(self.episodic_counting_memory),
                copy.deepcopy(self.obj_centric_episodic_counting_memory))

    def restore_checkpoint(self, checkpoint):
        '''
        restore the environment to a previous state
        '''
        # Set environment to previous
        self.env.close()
        self.env = self.env_stack.pop()

        self.state_history, self.action_history, self.episodic_counting_memory, self.obj_centric_episodic_counting_memory = checkpoint

    def step(self, action):
        next_state, reward, done, infos = self.env.step([action])

        curr_reward = self._get_current_reward(reward[0], next_state)

        next_state = self._map_state(next_state, infos)
        self.state_history.append(next_state)
        self.action_history.append(action)
        return next_state, curr_reward, done[0], None, infos

    def _get_current_reward(self, step_reward, next_state):
        MAX_NORM_VALUE = 0.2
        new_state_reward = self.episodic_counting_memory.is_a_new_state(next_state)[0] # Between 0 and 1
        new_state_reward = new_state_reward * MAX_NORM_VALUE # Normalize value to be between 0 and 0.2
        new_object_reward = self.obj_centric_episodic_counting_memory.get_object_novelty_reward(next_state)[0] # Between 0 and 1
        new_object_reward = new_object_reward * MAX_NORM_VALUE # Normalize value to be between 0 and 0.2
        step_reward = -0.5 if step_reward == 0 else 100 # Haven't reached goal: 0 -> -3, Reaching goal: 100

        current_reward = step_reward + new_state_reward + new_object_reward

        self.episodic_counting_memory.push(next_state)
        self.obj_centric_episodic_counting_memory.push(next_state)

        return current_reward
    
    def get_valid_actions(self, state):
        return state['valid_actions']
    
    def get_valid_actions_text(self, state):
        return state['valid_actions'] # Valid actions already in text
    
    def state_to_text(self, state):
        return state['text_state']
    
    def action_to_text(self, action):
        return action # Action already in text
    
    def action_txt_to_idx(self, action_txt, valid_actions_txt):
        return valid_actions_txt.index(action_txt)
    
    def get_state_and_action_history(self):
        return self.state_history, self.action_history

    def _load_config(self, config_path):
        with open(config_path) as reader:
            config = yaml.safe_load(reader)
        return config

    def _map_state(self, state, infos):
        return {
            'text_state': state[0],
            'valid_actions': infos['admissible_commands'][0],
        }
    
