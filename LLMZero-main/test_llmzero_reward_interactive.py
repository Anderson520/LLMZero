import matplotlib.pyplot as plt
import random
import numpy as np
from agents.elevator_expert import ElevatorExpertPolicyAgent
from agents.random_agent import RandomAgent
from agents.llmzero import LLMTransitionModel, LLMRewardModel
from environments.ElevatorEnvironment import ElevatorEnvironment
import sys
import os
import dotenv
dotenv.load_dotenv()

sys.path.append('..')


env = ElevatorEnvironment()

llmzero_reward_model = LLMRewardModel(
    debug=True,
    env_params={
        "system_prompt_path": "prompts/prompt_elevator_reward.txt",
        "extract_reward_regex": r"TOTAL_REWARD_FINAL = (.*)\n", # only use the first match, same line
        "extract_reward_regex_fallback": [r"TOTAL_REWARD_FINAL = (.*)\n"],
    }

)

num_steps = 0
state, _ = env.reset(42)
state_str = env.state_to_text(state)
print(state_str)

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    action = int(
        input("Enter action (0 = nothing, 1 = move, 2 = close door, 3 = open door): "))

    action_str = ""
    if action == 0:
        action_str = "nothing"
    elif action == 1:
        action_str = "move"
    elif action == 2:
        action_str = "close"
    elif action == 3:
        action_str = "open"

    llm_reward, status = llmzero_reward_model.get_reward(state_str, action_str)
    next_state, reward, done, _, info = env.step(action)
    state_str = env.state_to_text(next_state)
    print(state_str)
    print(f"Reward: {reward}")
    print(f"LLM reward: {llm_reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    if done:
        break

    num_steps += 1
