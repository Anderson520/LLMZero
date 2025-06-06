import os
import numpy as np
import pickle
import datetime
from agents.mcts import StateNode, ActionNode
from agents.llm_policy import LLMPolicyAgent
from utils import DictToObject, softmax
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import os
import re
import tqdm
import time

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# if os.getenv("USE_OPENAI_CUSTOM") == "True":
#     client = OpenAI(
#         base_url=os.getenv("CUSTOM_BASE_URL"),
#         api_key=os.getenv("CUSTOM_API_KEY")
#     )
# else:
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   
class LLMModel: 
        def __init__(self, 
                 env_params,
                 load_prompt_buffer_path=None,
                 prompt_buffer_prefix="prompt_buffer/elevator_reward",
                 save_buffer_interval=1,
                 client=None,
                 llm_model='gpt-4o-mini', 
                 debug=False):
            
            self.env_params = env_params
            
            self.system_prompt = open(self.env_params["system_prompt_path"], "r").read()
            self.client = client
            self.llm_model =  llm_model
            self.debug = debug
            
            self.prompt_buffer = {}
            if load_prompt_buffer_path is not None:
                #check file exists
                if os.path.exists(load_prompt_buffer_path):
                    with open(load_prompt_buffer_path, "rb") as f:
                        print(f"Loading prompt buffer from {load_prompt_buffer_path}")
                        self.prompt_buffer = pickle.load(f)        
                else:
                    print(f"Warning: Prompt buffer file {load_prompt_buffer_path} not found. Creating new buffer.")        
            
            self.prompt_buffer_save_path = f"{prompt_buffer_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.save_buffer_interval = save_buffer_interval
            self.call_count = 0
        
        def query_llm(self, user_prompt):
            '''
            Query the LLM with the user prompt
            '''
            
            if user_prompt in self.prompt_buffer:
                if self.debug:
                    print("Prompt found in buffer")
                    print(self.prompt_buffer[user_prompt])
                
                return self.prompt_buffer[user_prompt]
            
            # print("********************************************************")
            # print("user_prompt")
            # print(user_prompt)
            # print("********************************************************")
        
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
            
            while True:
                try:
                    response = self.client.chat.completions.create(model=self.llm_model, messages=messages)
                    break
                except Exception as e:
                    print(f"Error calling API: {e}, retrying...")
                    time.sleep(1)
            
            # grab the content of the first choice (only one choice is returned)
            # print("********************************************************")
            # print("response\n", response)
            # print("********************************************************")
            
            response = response.choices[0].message.content
            
            self.prompt_buffer[user_prompt] = response
            
            self.call_count += 1
            
            if self.call_count % self.save_buffer_interval == 0:
                self.save_prompt_buffer(self.prompt_buffer_save_path)
            
            if self.debug:
                print(f"response:\n {response}")
            
            return response
        
        def save_prompt_buffer(self, save_path):
            try:
                with open(save_path, "wb") as f:
                    pickle.dump(self.prompt_buffer, f)
            except Exception as e:
                print(f"Error saving prompt buffer: {e}")
        
    
    
class LLMTransitionModel(LLMModel):
    def __init__(self, 
                 env_params,
                 load_prompt_buffer_path=None,
                 prompt_buffer_prefix="prompt_buffer/elevator_transition",
                 save_buffer_interval=1,
                 llm_model='gpt-4o-mini', 
                 debug=False):
        
        client = OpenAI(
            base_url=os.getenv("CUSTOM_BASE_URL"),
            api_key=os.getenv("CUSTOM_API_KEY")
        )
        
        super().__init__(env_params, load_prompt_buffer_path, prompt_buffer_prefix, save_buffer_interval, client, 'open-mixtral-8x22b', debug) 
        
        self.extract_state_regex = env_params["extract_state_regex"]
        self.extract_state_regex_fallback = env_params["extract_state_regex_fallback"]        
        
    def get_next_state(self, state: str, action: str):
        '''
        Get the next state given the current state and action
        '''
        
        # construct user prompt
        user_prompt = "**Current State:**\n"
        user_prompt += state
        user_prompt += "\n**Action:**"
        user_prompt += action + "\n"
        
        response = self.query_llm(user_prompt)
        
        next_state, status = self.extract_state(response)
        
        return next_state, status
    
    def extract_state(self, response: str):
        '''
        Extract the next state from the LLM response
        '''
        
        match = re.search(self.extract_state_regex, response, re.DOTALL | re.IGNORECASE)
        if match is not None:
            next_state = match.group(1)
            return next_state, "success"
        else:
            if self.debug:
                print("Warning: No match found, trying fallback regex...")
            
            for regex in self.extract_state_regex_fallback:
                match = re.search(regex, response, re.DOTALL | re.IGNORECASE)
                if match is not None:
                    next_state = match.group(1)
                    return next_state, "success on fallback regex"
            else:
                print("Error: No match found with fallback regex, using full response as next state")
                return response, "error"
        
class LLMRewardModel(LLMModel):
        def __init__(self, 
                 env_params,
                 load_prompt_buffer_path=None,
                 prompt_buffer_prefix="prompt_buffer/elevator_reward",
                 save_buffer_interval=1,
                 llm_model='gpt-4o-mini', 
                 debug=False):
            
            # client = OpenAI(
            #     base_url=os.getenv("CUSTOM_BASE_URL"),
            #     api_key=os.getenv("CUSTOM_API_KEY")
            # )
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            llm_model = 'gpt-4o-mini'
            
            super().__init__(env_params, load_prompt_buffer_path, prompt_buffer_prefix, save_buffer_interval, client, llm_model, debug)
            
            self.reward_regex = env_params["extract_reward_regex"]
            self.reward_regex_fallback = env_params["extract_reward_regex_fallback"]
            self.extract_done_regex = env_params["extract_done_regex"]
            self.extract_done_regex_fallback = env_params["extract_done_regex_fallback"]
            
        def get_reward_done(self, state: str, action: str):
            '''
            Get the reward given the current state and action
            '''
            
            # construct user prompt
            user_prompt = state
            
            response = self.query_llm(user_prompt)
            
            reward, status_reward = self.extract_reward(response)
            
            done, status_done = self.extract_done(response)
            
            status = f"Reward: {status_reward}, Done: {status_done}"
            
            return reward, done, status
            
        def extract_reward(self, response: str):
            # '''
            # Extract the reward from the LLM response
            # '''
            match = re.search(self.reward_regex, response)
            if match is not None:
                reward = match.group(1)
                return float(reward), "success"
            else:
                if self.debug:
                    print("Warning: No match found, trying fallback regex...")
                
                for regex in self.reward_regex_fallback:
                    match = re.search(regex, response)
                    if match is not None:
                        reward = match.group(1)
                        return float(reward), "success on fallback regex"
                else:
                    #try to extract the reward as the last float in the response
                    matches = re.findall(r"-?\d+(?:\.\d+)?", response)
                    
                    if len(matches) > 0:
                        reward = matches[-1]
                        return float(reward), "success on last float"
                    
                    print("Error: No match found with fallback regex, returning 0 as reward")
                    return 0, "error"
                
        def extract_done(self, response: str):
            '''
            Extract the done flag from the LLM response
            '''
            def text_to_bool(text):
                if "true" in text.lower():
                    return True
                # return False by default
                return False
            
            match = re.search(self.extract_done_regex, response, re.DOTALL | re.IGNORECASE)
            if match is not None:
                done = match.group(1)
                return text_to_bool(done), "success"
            else:
                # if self.debug:
                #     print("Warning: No match found, trying fallback regex...")
                
                for regex in self.extract_done_regex_fallback:
                    match = re.search(regex, response, re.DOTALL | re.IGNORECASE)
                    if match is not None:
                        done = match.group(1)
                        return text_to_bool(done), "success on fallback regex"
                else:
                    # print("Error: No match found with fallback regex, using full response as done")
                    return False, "error"
                
class LLMValueModel(LLMModel):
    def __init__(self,
                    env_params,
                    load_prompt_buffer_path=None,
                    prompt_buffer_prefix="prompt_buffer/elevator_value",
                    save_buffer_interval=1,
                    llm_model='gpt-4o-mini',
                    debug=False):
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))            
            
        super().__init__(env_params, load_prompt_buffer_path, prompt_buffer_prefix, save_buffer_interval, client, "gpt-4o", debug)
            
        self.extract_value_regex = env_params["extract_value_regex"]
        self.extract_value_regex_fallback = env_params["extract_value_regex_fallback"]
            
    def get_value(self, state: str):
        '''
        Get the value of the state
        '''
        
        # construct user prompt
        user_prompt = "**State:**\n"
        user_prompt += state
        
        response = self.query_llm(user_prompt)
        
        value, status = self.extract_value(response)
        
        return value, status
    
    def extract_value(self, response: str):
        '''
        Extract the value from the LLM response
        '''
        
        match = re.search(self.extract_value_regex, response, re.DOTALL | re.IGNORECASE)
        if match is not None:
            value = match.group(1)
            return float(value), "success"
        else:
            if self.debug:
                print("Warning: No match found, trying fallback regex...")
            
            for regex in self.extract_value_regex_fallback:
                match = re.search(regex, response, re.DOTALL | re.IGNORECASE)
                if match is not None:
                    value = match.group(1)
                    return float(value), "success on fallback regex"
            else:
                #try to extract the reward as the last float in the response
                matches = re.findall(r"-?\d+(?:\.\d+)?", response)
                
                if len(matches) > 0:
                    reward = matches[-1]
                    return float(reward), "success on last float"
                    
                print("Error: No match found with fallback regex, using 0 as value")
                return 0, "error"
            
                 
                
class LLMZeroAgent:
    def __init__(self, env, cfg=None, debug=False):
        
        self.env = env  # env object is only needed for the state_to_text, and get_valid_actions methods, no steps needed
        
        self.cfg = {
            # "env": "elevator",  # TO DO: implement for alfworld
            "mcts": {
                "num_simulations": 50,
                "c_puct": 100,    #should be proportional to the scale of the rewards 
                "gamma": 0.99,
                "max_depth": 100,   # setting this higher would have less of an effect because there is no rollout
                "backprop_T": 50,
            },
            "llm_policy": {
                "env_params": {
                    "system_prompt_path": "prompts/prompt_elevator_policy.txt",
                    "extract_action_regex": r"optimal action: (.*)",
                },
                "load_prompt_buffer_path": "prompt_buffer/elevator_policy_llmzero.pkl", # update this path to the path of the saved prompt buffer
                "prompt_buffer_prefix": "prompt_buffer/elevator_policy",
                "save_buffer_interval": 5,
            } ,
            "llm_transition": {
                "env_params": {
                    "system_prompt_path": "prompts/prompt_elevator_transition_2.txt",
                    "extract_state_regex": r"next state:(.*?)```",
                    "extract_state_regex_fallback": [r"next state:(.*)", r"```plaintext(.*)```", r"\*\*Next State\*\*:\n(.*)"],
                },
                "load_prompt_buffer_path": "prompt_buffer/elevator_transition_llmzero.pkl", # update this path to the path of the saved prompt buffer   
            },
            "llm_reward": {
                "env_params": {
                    "system_prompt_path": "prompts/prompt_elevator_reward.txt",
                    "extract_reward_regex": r"TOTAL_REWARD_FINAL = (-?\d+.\d+)", # only use the first match, same line
                    "extract_reward_regex_fallback": [],
                    "extract_done_regex": r"done: (.*)",
                    "extract_done_regex_fallback": [r"done: (.*)"],
                },
                "load_prompt_buffer_path": "prompt_buffer/elevator_reward_llmzero.pkl",
            },
            "llm_value": {
                "env_params": {
                    "system_prompt_path": "prompts/prompt_elevator_value.txt",
                    "extract_value_regex": r"\\boxed\{(-?\d*\.?\d+)\}",
                    "extract_value_regex_fallback": [],
                },
                "load_prompt_buffer_path": "prompt_buffer/elevator_value_llmzero.pkl",
            }
        }
        
        if cfg is not None:
            self.cfg.update(cfg)
            
        self.args = self.cfg["mcts"]
        self.args = DictToObject(self.args)
            
        self.debug = debug
        
        # initialize policy
        self.policy = LLMPolicyAgent(env, device="cuda", debug=debug, **self.cfg["llm_policy"])
        self.transition_model = LLMTransitionModel(**self.cfg["llm_transition"], debug=debug)
        self.reward_model = LLMRewardModel(**self.cfg["llm_reward"], debug=debug)
        self.value_model = LLMValueModel(**self.cfg["llm_value"], debug=debug)
        
    def act(self, state):
        '''
        Run the MCTS algorithm to select the best action
        '''
        
        state = self.env.state_to_text(state)
        root = self.build_state(state)
        
        for _ in tqdm.tqdm(range(self.args.num_simulations)):
            self.simulate(root)
            
        best_action_text = self.select_action_node_greedily(root).action
        
        best_action = self.env.action_txt_to_idx(best_action_text)
        
        return best_action
            
        
    def simulate(self, state_node):
        '''
        Simulate a trajectory from the current state node
        '''
        
        current_state_node = state_node
        depth = 0
        
        # Step 1: Selection, traverse down the tree until a leaf node is reached
        while not current_state_node.done and depth < self.args.max_depth:
            
            best_action_node = self.select_action_node(state_node)
            # obs, reward, done, _, _ = self.env.step(best_action_node.action)
            obs, _ = self.transition_model.get_next_state(current_state_node.state, best_action_node.action)
            reward, done, _ = self.reward_model.get_reward_done(current_state_node.state, best_action_node.action)
            
            reward = reward * self.args.gamma
            next_state_node = self.build_state(obs, reward, done, current_state_node)
            next_state_id = next_state_node.get_unique_id()
            
            if next_state_id not in best_action_node.children.keys():
                # Step 2: Expansion, add the new state node to the tree
                best_action_node.children[next_state_id] = next_state_node
                break
            else:
                current_state_node = best_action_node.children[next_state_id]
                current_state_node.reward = (current_state_node.reward * current_state_node.N + reward) / (current_state_node.N + 1)
                depth += 1
                
        # Step 3: Get value of the leaf node using from LLM value model
        value, _ = self.value_model.get_value(current_state_node.state)
            
        # Step 4: Backpropagation, update the Q values of the nodes in the trajectory
        current_action_node = best_action_node
        cumulative_reward = value
        
        while current_action_node is not None:
            current_action_node.N += 1
            # current_action_node.Q += (cumulative_reward - current_action_node.Q) / current_action_node.N
            current_action_node.Rs.append(cumulative_reward)
            # softmax to prioritize actions with higher rewards
            # print("current_action_node.Rs", current_action_node.Rs)
            best_action_node.Q = np.sum(np.array(best_action_node.Rs) * softmax(best_action_node.Rs, T=self.args.backprop_T))
            current_state_node = current_action_node.parent
            current_state_node.N += 1
            cumulative_reward = current_state_node.reward + self.args.gamma * cumulative_reward
            current_action_node = current_state_node.parent
            
        return cumulative_reward    #return not actually needed, just for debugging
            
            
    def build_state(self, state, reward=0, done=False, parent=None):
        valid_actions = self.env.get_valid_actions_text(state)  # using text instead of idx
        state_node = StateNode(state, valid_actions, reward, done, parent)
        if self.policy is not None:
            # print("state", state)
            distribution = self.policy.get_action_distribution(state)
            if isinstance(distribution, dict):
                distribution = list(distribution.values())
            state_node.children_probs = distribution
        
        return state_node
        
        
    def select_action_node(self, state_node, debug=False):
        '''
        Select the action with the highest UCB value
        '''
        
        best_ucb = -np.inf
        best_children = []
        best_children_prob = []
        EPS = 1e-6
        
        for i in range(len(state_node.children)):
            child = state_node.children[i]
            child_prob = state_node.children_probs[i]
            
            #PUCT formula
            c_puct = self.args.c_puct * len(state_node.children) # multiply to offset child_prob's effect
            ucb = child.Q + c_puct * child_prob * np.sqrt(np.log(state_node.N) / (1 + child.N))
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_children = [child]
                best_children_prob = [child_prob]
            elif ucb == best_ucb:
                best_children.append(child)
                best_children_prob.append(child_prob)
                
        if debug:
            for i, child in enumerate(state_node.children):
                if child.N > 0:
                    print(f"Action {child.action}: Q = {child.Q}, N = {child.N}, prob = {state_node.children_probs[i]}")
                    
        best_children_prob = np.array(best_children_prob) / (np.sum(best_children_prob)+EPS)
        best_action_idx = np.argmax(best_children_prob)
        
        return best_children[best_action_idx]
    
    def select_action_node_greedily(self, state_node):
        '''
        Select the action with the most visits
        '''
        
        best_children = []
        best_children_prob = []
        most_visits = 0
        
        for i, child in enumerate(state_node.children):
            # if self.debug:
            if True:
                print(f"Action {child.action}: Q = {child.Q}, N = {child.N}")
            
            if child.N == most_visits:
                most_visits = child.N
                best_children.append(child)
                best_children_prob.append(state_node.children_probs[i] + child.Q)
            if child.N > most_visits:
                most_visits = child.N
                best_children = [child]
                best_children_prob = [state_node.children_probs[i] + child.Q]
                
        # in case of ties, return highest Q value + child prob
        best_children_prob = np.array(best_children_prob)
        best_child = best_children[np.argmax(best_children_prob)]
        best_action = best_child
                    
        return best_action
    
    