import numpy as np
from utils import DictToObject, softmax, elevator_estimate_value
import tqdm
import json

# np.random.seed(42)

class StateNode:
    def __init__(self, state, valid_actions, reward = 0, done = False, parent=None):
        self.state = state
        self.valid_actions = valid_actions
        self.reward = reward
        self.done = done
        self.parent = parent   # link back to previous action node
        
        self.children = []
        for action in valid_actions:
            self.children.append(ActionNode(action, self))
        
        self.children_probs = np.ones(len(valid_actions)) / len(valid_actions)    
        
        self.N = 1 # number of times this state node has been visited, node created implies one visit
        
    def get_unique_id(self):
        #convert state to a unique string
        if isinstance(self.state, dict):
            data = {key: (value.tolist() if isinstance(value, np.ndarray) 
                        else int(value) if isinstance(value, np.integer) 
                        else float(value) if isinstance(value, np.floating) 
                        else bool(value) if isinstance(value, np.bool_)
                            else value)
            for key, value in self.state.items()}
        elif isinstance(self.state, str):
            return self.state
        else:
            data = self.state.tolist() if isinstance(self.state, np.ndarray) else self.state
        return json.dumps(data)

class ActionNode:
    '''
    Action nodes are like chance nodes in MDPs, they represent the possible outcomes of taking an action
    '''
    def __init__(self, action, parent):
        self.action = action
        self.parent = parent # link back to previous state node
        self.N = 0 # number of times this action node has been visited
        self.Q = 0 # average reward of this action node
        self.Rs = [] # list of rewards received from this action node
        self.children = {}  # dictionary of next state nodes id: StateNode
        
class MCTSAgent:
    def __init__(self,
                 env,
                 policy=None,
                 args=None,
                 debug=False):
        '''
        env: pyRDDLGym environment object, must be discrete action space
        policy: object, policy to use for selecting actions
        args: dict, additional arguments to override default values
        '''
        
        self.env = env
        self.policy = policy
        
        self.args = {
            "num_simulations": 200,
            "c_puct": 100,    #should be proportional to the scale of the rewards 
            "gamma": 1,
            "max_depth": 40,
            "num_rollouts": 10,
            "backprop_T": 50,
        }
        
        if args is not None:
            self.args.update(args)
            
        self.args = DictToObject(self.args)
            
        self.debug = debug
        
    def act(self, state):
        '''
        Run the MCTS algorithm to select the best action
        '''
        
        root = self.build_state(state)
        
        for _ in tqdm.tqdm(range(self.args.num_simulations)):
            checkpoint = self.env.checkpoint()
            seed = np.random.randint(0, 1000)
            self.env.base_env.seed(seed)
            self.simulate(root)
            self.env.restore_checkpoint(checkpoint)
            
        best_action = self.select_action_node_greedily(root).action
        
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
            obs, reward, done, _, _ = self.env.step(best_action_node.action)
            
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
                
        # Step 3: Rollout, simulate the rest of the trajectory using a random policy
        rollout_rewards = []
        
        for _ in range(self.args.num_rollouts):
            checkpoint = self.env.checkpoint()
            rollout_reward = 0
            rollout_depth = 0
            tmp_depth = depth
            
            obs = current_state_node.state
            while not done and tmp_depth < self.args.max_depth:
                valid_actions = self.env.get_valid_actions(obs)
                action = np.random.choice(valid_actions)
                obs, reward, done, _, _ = self.env.step(action)
                tmp_depth += 1
                rollout_depth += 1
                rollout_reward += reward * self.args.gamma ** rollout_depth
                
            rollout_rewards.append(rollout_reward)
            self.env.restore_checkpoint(checkpoint)
            
        rollout_reward = np.mean(rollout_rewards)
            
        # test using value estimation instead of rollout
        # rollout_reward = elevator_estimate_value(next_state_node.state)
        # print("-----------------------------------------")
        # print(f"Action: {best_action_node.action}")
        # print(f"State:\n {self.env.state_to_text(next_state_node.state)}")
        # print(f"Rollout reward: {rollout_reward}")
        # print("-----------------------------------------")
            
        # Step 4: Backpropagation, update the Q values of the nodes in the trajectory
        current_action_node = best_action_node
        cumulative_reward = rollout_reward
        
        while current_action_node is not None:
            current_action_node.N += 1
            # current_action_node.Q += (cumulative_reward - current_action_node.Q) / current_action_node.N
            current_action_node.Rs.append(cumulative_reward)
            # softmax to prioritize actions with higher rewards
            # best_action_node.Q = np.sum(np.array(best_action_node.Rs) * softmax(best_action_node.Rs, T=self.args.backprop_T))
            best_action_node.Q = np.mean(best_action_node.Rs)
            current_state_node = current_action_node.parent
            current_state_node.N += 1
            cumulative_reward = current_state_node.reward + self.args.gamma * cumulative_reward
            current_action_node = current_state_node.parent
            
        # for i, child in enumerate(state_node.children):
        #     if self.debug:
        #         print(f"Action {child.action}: Q = {child.Q}, N = {child.N}")
        
        return cumulative_reward    #return not actually needed, just for debugging
            
            
    def build_state(self, state, reward=0, done=False, parent=None):
        valid_actions = self.env.get_valid_actions(state)
        state_node = StateNode(state, valid_actions, reward, done, parent)
        if self.policy is not None:
            distribution = self.policy.get_action_distribution(state)
            if isinstance(distribution, dict):
                distribution = list(distribution.values())
            state_node.children_probs = distribution
            # if self.debug:
            #     # print(f"State: {state}")
            #     print(f"Action distribution: {distribution}")
        
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
        
        
        
        