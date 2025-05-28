# import numpy as np
# import yaml
import inquirer
# import gym
# import random
import os
from enum import Enum
# import torch
from dotenv import load_dotenv, set_key, get_key

def create_env_file_if_needed():
    env_file = ".env"

    keys = [
        "OPENAI_API_KEY",
        "CUSTOM_BASE_URL",
        "CUSTOM_API_KEY",
        "CUSTOM_MODEL_ID",
        "USE_OPENAI_CUSTOM",
    ]
    
    print(f"Checking if {env_file} exists...")

    if not os.path.exists(env_file):
        print(f"{env_file} not found. Creating it...")
        open(env_file, 'w').close()

    load_dotenv(env_file)

    for key in keys:
        if get_key(dotenv_path=env_file, key_to_get=key) is None:
            print(f"{key} not found in {env_file}. Adding it...")
            set_key(env_file, key, "")
            print(f"{key} is added to {env_file}.")
        else:
            print(f"{key} already exists in {env_file}. No changes needed.")

# from environments.ALFWorldEnvironment import ALFWorldEnvironment
# from environments.ElevatorEnvironment import ElevatorEnvironment
# from agents.random_agent import RandomAgent
# from agents.nn_agent import NNAgent
# from agents.mcts import MCTSAgent
# from agents.llmzero import LLMZeroAgent
# from agents.llm_policy import LLMPolicyAgent
# from agents.elevator_expert import ElevatorExpertPolicyAgent

# from agents.alfworld_mcts import ALFworldMCTSAgent
# from agents.alfworld_llmzero import ALFWorldLLMZeroAgent
# from agents.alfworld_llm_policy import ALFWorldLLMPolicyAgent

# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)


class ENVIRONMENT(str, Enum):
    ELEVATOR = "elevator"
    ALF_WORLD = "alfWorld"

    @classmethod
    def list(self):
        return list(map(lambda c: c.value, self))


class AGENT(str, Enum):
    LLM_POLICY = "llm"
    RANDOM = "random"
    MCTS = "mcts"
    EXPERT = "expert"
    MCTS_EXPERT = "mcts-expert"
    LLM_MCTS = "mcts-llm"
    NEURAL_NETWORK = "nn"
    LLM_ZERO = "llmzero"

    @classmethod
    def list(self, env: ENVIRONMENT):
        if env == ENVIRONMENT.ELEVATOR:
            return list(map(lambda c: c.value, self))
        elif env == ENVIRONMENT.ALF_WORLD:
            return [AGENT.LLM_POLICY.value, AGENT.LLM_MCTS.value, AGENT.LLM_ZERO.value, AGENT.MCTS.value]

# def log(log_file, text):
#     log_file.write('\n')
#     log_file.write(text)
#     log_file.flush()
#     print(text)

# def load_env(env_str: ENVIRONMENT):
#     print("env_str", env_str)
#     if env_str == ENVIRONMENT.ELEVATOR.name:
#         env = ElevatorEnvironment()
#         cfg = yaml.safe_load(open("configs/elevator.yaml"))
#     elif env_str == ENVIRONMENT.ALF_WORLD.name:
#         env = ALFWorldEnvironment()
#         cfg = yaml.safe_load(open("configs/alfworld.yaml"))
#     else:
#         raise Exception("Invalid environment selected.")
#     return env, cfg


# def load_agent(agent_str, env_str,  env: gym.Wrapper, cfg):
#     if agent_str == AGENT.LLM_POLICY.name:
#         if env_str == ENVIRONMENT.ELEVATOR.name:
#             agent = LLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_policy"])
#         elif env_str == ENVIRONMENT.ALF_WORLD.name:
#             agent = ALFWorldLLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_policy"])
#         else:
#             raise ValueError("Invalid Environment")
#     elif agent_str == AGENT.RANDOM.name:
#         agent = RandomAgent(env, seed=SEED)
#     elif agent_str == AGENT.MCTS.name:
#         agent = MCTSAgent(env, policy=None, debug=True)
#     elif agent_str == AGENT.EXPERT.name:
#         agent = ElevatorExpertPolicyAgent()  # TO DO: abstract out the expert policy
#     elif agent_str == AGENT.MCTS_EXPERT.name:
#         mcts_args = cfg["mcts_expert"]["mcts_args"]
#         agent = MCTSAgent(env, policy=ElevatorExpertPolicyAgent(),
#                           debug=False, args=mcts_args)
#     elif agent_str == AGENT.LLM_MCTS.name:
#         mcts_args = cfg["llm_mcts"]["mcts_args"]
#         if env_str == ENVIRONMENT.ELEVATOR.name:
#             llm_agent = LLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_mcts"]["llm_policy"])
#             agent = MCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)
#         elif env_str == ENVIRONMENT.ALF_WORLD.name:
#             llm_agent = ALFWorldLLMPolicyAgent(env, device="cuda", debug=False, **cfg["llm_mcts"]["llm_policy"])
#             agent = ALFworldMCTSAgent(env, policy=llm_agent, debug=False, args=mcts_args)
#         else:
#             raise ValueError("Invalid Environment")
#     elif agent_str == AGENT.NEURAL_NETWORK.name:
#         agent = NNAgent(env, cfg=cfg["nn_agent"])
#     elif agent_str == AGENT.LLM_ZERO.name:
#         if env_str == ENVIRONMENT.ELEVATOR.name:
#             agent = LLMZeroAgent(env)
#         elif env_str == ENVIRONMENT.ALF_WORLD.name:
#             agent = ALFWorldLLMZeroAgent(env)
#         else:
#             raise ValueError("Invalid Environment")
#     else:
#         raise ValueError("Invalid agent")

#     return agent


# def user_selection():
#     env_question = [
#         inquirer.List("env", "Select an environment", ENVIRONMENT.list()),
#     ]
#     env_answer = inquirer.prompt(env_question)
    
#     agent_question = [
#         inquirer.List("agent", "Select an agent", AGENT.list(env=ENVIRONMENT(env_answer["env"]))),
#     ]
#     agent_answer = inquirer.prompt(agent_question)

#     selected_env = ENVIRONMENT(env_answer["env"]).name
#     selected_agent = AGENT(agent_answer["agent"]).name
    
#     # print('ENVIRONMENT(env_answer["env"]).name', ENVIRONMENT(env_answer["env"]).name)
#     # print('AGENT(agent_answer["agent"]).name', AGENT(agent_answer["agent"]).name)

#     return selected_env, selected_agent

# def run_alfworld(agent_str, agent, env):
#     if os.getenv("ALFWORLD_DATA") is None:
#         raise Exception("Must set path in ALFWORLD_DATA environment variable to use ALFWorld Environment")
    
#     LOGS_DIR_PATH = 'logs/'
#     FILES_TO_TEST = [
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-AlarmClock-None-Desk-307/trial_T20190907_072303_146844',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Book-None-Bed-312/trial_T20190908_103648_829231',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Bowl-None-Fridge-6/trial_T20190906_230933_751794',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Candle-None-Toilet-413/trial_T20190909_104025_525772',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-CellPhone-None-Bed-322/trial_T20190907_163932_211569',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-CreditCard-None-Shelf-316/trial_T20190909_092853_746076',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Laptop-None-Bed-302/trial_T20190908_112426_055221',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Pen-None-SideTable-309/trial_T20190907_051843_166835',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Pencil-None-SideTable-322/trial_T20190908_112624_358795',
#         '$ALFWORLD_DATA/json_2.1.1/train/pick_and_place_simple-Pillow-None-ArmChair-206/trial_T20190909_011649_821132'
#     ]
    
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     agent_name = agent_str

#     log_file = open(f'{LOGS_DIR_PATH}{agent_name}_trial_logs.log', 'a')

#     rewards = []
#     num_failure = 0
#     num_success = 0

#     for file in FILES_TO_TEST:
#         log(log_file, '--------------------------------------------------')

#         env = ALFWorldEnvironment(overwrite_env=file)
#         success, num_steps, total_reward = run_single_trial(env, agent, log_file)

#         if success:
#             num_success += 1
#             log(log_file, f'SUCCESS, Total steps: {num_steps}, total reward: {total_reward}')
#         else:
#             num_failure += 1
#             log(log_file, f'FAIL, Total steps: {num_steps}, total reward: {total_reward}')
    
#         rewards.append(total_reward)

#     log(log_file, f"Average reward: {np.mean(rewards)}")
#     log(log_file, f'Num SUCCESS: {num_success}')
#     log(log_file, f'Num FAIL: {num_failure}')

#     log_file.close()

# def run_elevator(agent_str, agent, env): 
#     rewards = []
    
#     for i in range(1):
#         num_steps = 0
#         total_reward = 0
#         if isinstance(agent, RandomAgent) or isinstance(agent, NNAgent):
#             state, _ = env.reset(seed=SEED)
#         else:
#             state, _ = env.reset()

#         while True:
#             action = agent.act(state)
#             print(env.action_to_text(action))
#             next_state, reward, done, _, info = env.step(action)

#             total_reward += reward

#             if done:
#                 rewards.append(total_reward)
#                 break

#             state = next_state
#             num_steps += 1

#             print(
#                 f"Step {num_steps}, action: {env.action_to_text(action)}, reward: {reward}, total reward: {total_reward}")

#     print("Average reward: ", np.mean(rewards))

# def run_single_trial(env, agent, log_file):
#     num_steps = 0
#     total_reward = 0
#     max_num_steps = 40

#     state, _ = env.reset()
#     log(log_file, state['text_state'])
#     while True:
#         action = agent.act(state)
#         print(action)
#         next_state, reward, done, _, info = env.step(action)

#         num_steps += 1
#         total_reward += reward
#         print(f"Step {num_steps}, action: {env.action_to_text(action)}, reward: {reward}, total reward: {total_reward}")
        
#         if done:
#             break
#         if num_steps >= max_num_steps:
#             print('Unable to finish task')
#             break
        
#         state = next_state
    
#     return done, num_steps, total_reward


# def run():
    # selected_env, selected_agent = user_selection()

    # env, cfg = load_env(selected_env)
    # agent = load_agent(selected_agent, selected_env, env, cfg)
    
    # if selected_env == ENVIRONMENT.ELEVATOR.name:
    #     run_elevator(selected_agent, agent, env)
    # elif selected_env == ENVIRONMENT.ALF_WORLD.name:
    #     run_alfworld(selected_agent, agent, env)
    

if __name__ == "__main__":
    # check if .env file exist, if not create one and append something to it
    # run()
    create_env_file_if_needed()
    os.system("source .env")

    env_question = [
        inquirer.List("env", "Select an environment", ENVIRONMENT.list()),
    ]
    env_answer = inquirer.prompt(env_question)
    
    agent_question = [
        inquirer.List("agent", "Select an agent", AGENT.list(env=ENVIRONMENT(env_answer["env"]))),
    ]
    agent_answer = inquirer.prompt(agent_question)
    
    if env_answer.get("env") == ENVIRONMENT.ELEVATOR.value:
        os.system(f"python test.py --agent {agent_answer.get('agent')} --seed 42")
    elif env_answer.get("env") == ENVIRONMENT.ALF_WORLD.value:
        os.system(f"python test_alfworld.py --agent {agent_answer.get('agent')} --seed 42")
