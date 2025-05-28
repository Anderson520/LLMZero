import numpy as np
import os
from openai import OpenAI
import time
import json

from agents.llm_policy import LLMPolicyAgent

# Only supports OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ALFWorldLLMPolicyAgent(LLMPolicyAgent):
    def __init__(self, 
                 env,
                 device, 
                 llm_model='gpt-4o-mini', 
                 env_params=None,
                 api_params=None,
                 load_prompt_buffer_path=None,
                 prompt_buffer_prefix="prompt_buffer/alfworld",
                 save_buffer_interval=1,
                 overwrite_prompt_buffer=False,
                 debug=False,
                 temp=0.3   #smoothing factor for action distribution
                ):

        custom_env_params = {
            "system_prompt_path": "prompts/prompt_alfworld_policy.txt",
        }
        if env_params is not None:
            custom_env_params.update(env_params)

        custom_api_params = {
            "n": 1,
        }
        if api_params is not None:
            custom_api_params.update(api_params)

        super().__init__(env, device, llm_model, custom_env_params, custom_api_params, load_prompt_buffer_path, prompt_buffer_prefix, save_buffer_interval, debug, temp)

        if overwrite_prompt_buffer and load_prompt_buffer_path is not None:
            self.prompt_buffer_save_path = load_prompt_buffer_path

    def act(self, state, greedy=True):
        int_action = super().act(state, greedy)
        text_action = state['valid_actions'][int_action]
        return text_action
    
    def get_action_distribution(self, state, state_history=None, action_history=None):
        if state_history is None or action_history is None:
            state_history, action_history = self.env.get_state_and_action_history()

        messages = self._build_llm_messages(state, state_history, action_history)

        return_msgs = self.query_llm(messages)

        valid_actions_text = self.env.get_valid_actions_text(state)
        dist = self._get_action_distribution(return_msgs, valid_actions_text)
            
        if self.debug:
            state_text = self.env.state_to_text(state)
            print(f"State: {state_text}")
            print(f"Action distribution: {dist}")
            
        return dist

    def _build_llm_messages(self, curr_state, state_history, action_history):
        def state_to_text(state):
            state_text = self.env.state_to_text(state)
            valid_actions_text = self.env.get_valid_actions_text(state)

            text = "**State**: "
            text += state_text
            text += "\n**Valid Actions**: "
            text += ', '.join(valid_actions_text)
            return text

        messages = [{"role": "system", "content": self.system_prompt}]

        for state, action in zip(state_history[:-1], action_history):
            messages.extend([
                {"role": "user", "content": state_to_text(state)},
                {"role": "assistant", "content": f'Optimal action: {action}'},
            ])

        messages.append({"role": "user", "content": state_to_text(curr_state)})

        return messages

    def query_llm(self, messages):
        buffer_key = json.dumps(messages)
        if buffer_key in self.prompt_buffer:
            outputs = self.prompt_buffer[buffer_key]
            return outputs

        while True:
            try:
                response = client.chat.completions.create(model=self.llm_model, messages=messages, **self.api_params)
                break
            except Exception as e:
                print(f"Error calling API: {e}, retrying...")
                time.sleep(1)

        primary_response = response.choices[0].message.content # Only get main response
        if self.debug:
            print(f"Primary response (CUSTOM): {primary_response}")
        return_msgs = [primary_response]

        # Cache the generated responses and probabilities
        self.prompt_buffer[buffer_key] = return_msgs
        self.call_count += 1

        # Save periodically
        if self.call_count % self.save_buffer_interval == 0:
            self.save_prompt_buffer(self.prompt_buffer_save_path)

        return return_msgs

    def _get_action_distribution(self, messages, valid_actions_text):
        action_samples = [self.extract_action_from_message(message) for message in messages]

        cos_sim = self.compute_cos_sim(action_samples, valid_actions_text)

        action_probs = cos_sim.sum(axis=0)
        action_probs = np.clip(action_probs, 0, 1) # Handle cases when sum slightly exceeds 1
        action_probs = action_probs ** (1/self.temp)
        action_probs /= np.sum(action_probs + 1e-10)
        
        return action_probs

