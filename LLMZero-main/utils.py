import numpy as np
np.random.seed(42)

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

import matplotlib.pyplot as plt
from IPython.display import clear_output


def exponential_smoothing(data, alpha=0.1):
    """Compute exponential smoothing."""
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        st = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(st)
    return smoothed


def live_plot(data_dict):
    """Plot the live graph with multiple subplots."""

    plt.style.use('ggplot')
    n_plots = len(data_dict)
    fig, axes = plt.subplots(nrows=n_plots, figsize=(7, 4 * n_plots), squeeze=False)
    plt.subplots_adjust(hspace=0.5)
    plt.ion()
    clear_output(wait=True)

    for ax, (label, data) in zip(axes.flatten(), data_dict.items()):
        ax.clear()
        ax.plot(data, label=label, color="yellow", linestyle='--')
        # Compute and plot moving average for total reward
        if len(data) > 0:
            ma = exponential_smoothing(data)
            ma_idx_start = len(data) - len(ma)
            ax.plot(range(ma_idx_start, len(data)), ma, label="Smoothed Value",
                    linestyle="-", color="purple", linewidth=2)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc='upper left')

    plt.show()

# def elevator_estimate_value(state):
#     """Estimate the value of the state in the elevator environment."""
#     num_person_waiting = [None for _ in range(5)]
#     num_in_elavator = None
#     door_state = None
#     direction = None
#     current_floor = None
    
#     for feature, value in state.items():
#         if "num-person-waiting" in feature:
#             num_person_waiting[int(feature[-1])] = value
#         if "elevator-at-floor" in feature and value == True:
#             current_floor = int(feature[-1]) + 1
#         if feature == "elevator-dir-up___e0":
#             direction = "up" if value == True else "down"
#         if feature == "elevator-closed___e0":
#             door_state = "closed" if value == True else "open"
#         if feature == "num-person-in-elevator___e0":
#             num_in_elavator = value
            
#     total_waiting = sum(num_person_waiting)
            
#     #special cases
#     # no one is waiting, return -5 + current_floor
#     if total_waiting == 0:
#         return 0
    
#     # elevator at floor 1 with at least 1 person in elevator, return 30 * num_in_elevator - 3 * sum(num_person_waiting)
#     if current_floor == 1 and num_in_elavator > 0:
#         return 30 * num_in_elavator - 3 * total_waiting
    
#     reward = - 3 * sum(num_person_waiting) 
    
#     value = reward  # initial reward
    
#     if direction == "up":
#         # moving up to the floor takes floors_to_go moves, each move has the same reward
#         top_floor = 1
#         for i in range(5):
#             if num_person_waiting[i] > 0:
#                 top_floor = i + 1
                
#         floors_to_go = top_floor - current_floor
        
#         value += floors_to_go * reward
#         # current_floor = top_floor
        
#     reward -= 0.75  # penalty for moving down
        
#     cum_reward = 0
    
#     for i in range(current_floor -1, -1, -1):
#         reward += 3 * num_person_waiting[i]
#         if num_person_waiting[i] > 0:
#             cum_reward += 3 * reward - 3 * num_person_waiting[i]
#         else:
#             cum_reward += reward   
    
#     # # calculate delievery reward
#     waiting_below = sum(num_person_waiting[:current_floor])
#     delievered = min(10, num_in_elavator + waiting_below)
        
#     value += 30 * delievered
        
#     return value

def elevator_estimate_value(state):
    """Estimate the value of the state in the elevator environment."""
    # print(state)
    num_person_waiting = [None for _ in range(5)]
    num_in_elavator = None
    door_state = None
    direction = None
    current_floor = None
    
    for feature, value in state.items():
        if "num-person-waiting" in feature:
            num_person_waiting[int(feature[-1])] = value
        if "elevator-at-floor" in feature and value == True:
            current_floor = int(feature[-1]) + 1
        if feature == "elevator-dir-up___e0":
            direction = "up" if value == True else "down"
        if feature == "elevator-closed___e0":
            door_state = "closed" if value == True else "open"
        if feature == "num-person-in-elevator___e0":
            num_in_elavator = value
            
    value = 0
    
    if num_in_elavator == 10:
        return 300 - 3 * sum(num_person_waiting) - 0.75 * (current_floor - 1)
    
    # penalize for every passenger waiting above the current floor
    for i in range(current_floor, 5):
        floor_diff = i - current_floor
        value -= 3 * num_person_waiting[i - 1] * floor_diff
        
    # reward for every passenger waiting below or at the current floor
    # for i in range(0, current_floor):
    #     floor_diff = current_floor - i
    #     value += 30 * num_person_waiting[i] - 3 * floor_diff 
        
    _num_in_elavator = num_in_elavator
    
    
    for i in range(current_floor, 1, -1):
        remaining_capacity = 10 - _num_in_elavator
        if num_person_waiting[i - 1] > 0 and remaining_capacity > 0:
            value += (30 - 3 * (current_floor + 1 - i)) * min(num_person_waiting[i - 1], remaining_capacity) 
            _num_in_elavator += num_person_waiting[i - 1]
            _num_in_elavator = min(_num_in_elavator, 10)
        
    door_open = 1 if door_state == "open" else 0
    
    value = value - 3 * num_person_waiting[current_floor - 1] * (1 - door_open)
        
    # reward for every passenger in the elevator
    value += 30 * num_in_elavator
    penalty = 0.75 * (current_floor - 1) if num_in_elavator > 0 else 0
    value -= penalty
    
    # print(value)
    
    return value

def elevator_estimate_value(state):
    """Estimate the value of the state in the elevator environment."""
    # print(state)
    num_person_waiting = [None for _ in range(5)]
    num_in_elavator = None
    door_state = None
    direction = None
    current_floor = None
    
    for feature, value in state.items():
        if "num-person-waiting" in feature:
            num_person_waiting[int(feature[-1])] = value
        if "elevator-at-floor" in feature and value == True:
            current_floor = int(feature[-1]) + 1
        if feature == "elevator-dir-up___e0":
            direction = "up" if value == True else "down"
        if feature == "elevator-closed___e0":
            door_state = "closed" if value == True else "open"
        if feature == "num-person-in-elevator___e0":
            num_in_elavator = value
            
    delivered = num_in_elavator if current_floor == 1 else 0
    total_waiting = sum(num_person_waiting)
    is_people_in_elevator = num_in_elavator > 0
    
    reward = 30 * delivered - 3 * total_waiting - 0.75 * is_people_in_elevator
    
    value = reward
    
    if direction == "up":
        top_floor = 1
        for i in range(5):
            if num_person_waiting[i] > 0:
                top_floor = i + 1
                
        floors_to_go = top_floor - current_floor
        
        value += floors_to_go * reward
        
        current_floor = top_floor
        
    # open the door if there are passengers on the current floor
    if door_state == "closed" and num_person_waiting[current_floor - 1] > 0:
        # open the door
        value += reward
        door_state = "open"
        
    if door_state == "open":        
        # close the door
        total_waiting -= num_person_waiting[current_floor - 1]
        num_in_elavator += num_person_waiting[current_floor - 1]
        num_in_elavator = min(num_in_elavator, 10)
        is_people_in_elevator = num_in_elavator > 0
        reward = 30 * delivered - 3 * total_waiting - 0.75 * is_people_in_elevator
        
    # move down
    value += reward
    current_floor -= 1
    
    
    while current_floor > 1:
        if num_person_waiting[current_floor - 1] > 0 and num_in_elavator < 10:
            # open the door
            value += reward
            # close the door
            total_waiting -= num_person_waiting[current_floor - 1]
            num_in_elavator += num_person_waiting[current_floor - 1]
            num_in_elavator = min(num_in_elavator, 10)
            is_people_in_elevator = num_in_elavator > 0
            reward = 30 * delivered - 3 * total_waiting - 0.75 * is_people_in_elevator
            value += reward
            
        # move down
        value += reward
        current_floor -= 1
        
    # calculate delievery reward
    delivered = num_in_elavator
    
    value += 30 * delivered - 3 * total_waiting - 0.75 * is_people_in_elevator
    
    return value
            
    
    
    
    