import numpy as np
from agents.llm_policy import LLMPolicyAgent
# from agents.alfworld_llm_policy import ALFWorldLLMPolicyAgent
from agents.mcts import MCTSAgent
from agents.elevator_expert import ElevatorExpertPolicyAgent
from environments.ElevatorEnvironment import ElevatorEnvironment
from environments.BlackjackEnvironment import BlackjackEnvironment
# from environments.ALFWorldEnvironment import ALFWorldEnvironment

env = ElevatorEnvironment()
# env = BlackjackEnvironment()
# env = ALFWorldEnvironment()
env_params = {
            "system_prompt_path": "prompts/prompt_elevator_policy.txt",
            "extract_action_regex": r"optimal action: (.*)",
        }
# agent = LLMPolicyAgent(env, device="cuda", debug=True, temp=1.0, env_params=env_params)
# agent = ALFWorldLLMPolicyAgent(env, device="cuda", debug=True, temp=1.0, env_params=env_params)
# agent = ElevatorExpertPolicyAgent()
mcts_args = {
            "num_simulations": 100,
            "c_puct": 20,    #should be proportional to the scale of the rewards 
            "gamma": 0.95,
            "max_depth": 100,
            "num_rollouts": 10,
            "backprop_T": 50,
        }
# agent = MCTSAgent(env, policy=llm_agent, debug=True, args=mcts_args)
agent = MCTSAgent(env, policy=None, debug=True, args=mcts_args)

state, _ = env.reset()

num_steps = 0

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    action = agent.act(state)
    
    state, reward, done, _, _ = env.step(action)
    
    print(f"action: {env.action_to_text(action)}")
    print(f"state:\n{env.state_to_text(state)}")
    print("reward: ", reward)
    
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    
    wait = input("Press Enter to continue...")