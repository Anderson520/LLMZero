# from environments.ALFWorldEnvironment import ALFWorldEnvironment
from environments.ElevatorEnvironment import ElevatorEnvironment 
import numpy as np

env = ElevatorEnvironment()
ALFWorldEnvironment = None
# env = ALFWorldEnvironment()

    
# interactive, wait for user input
state, _ = env.reset()

num_steps = 0

print("Initial state:")
print(state)
print(env.state_to_text(state))

while True:
    print(f"---------------------- Step {num_steps} ----------------------")
    
    if type(env) == ALFWorldEnvironment:
        action = input("Enter action: ")
    else:
        action = input("Enter action: ")
        
    action = env.action_txt_to_idx(action)
    # print(f"Action: {env.action_to_text(action)}")
    next_state, reward, done, _, info = env.step(action)
    print("Next state:")
    # print(next_state)
    print(env.state_to_text(next_state))
    print(f"Reward: {reward}")
    if done:
        print("Episode done.")
        break
    
    num_steps += 1
    