# LLMZero

### Abstract
Deep Reinforcement Learning (DRL) has achieved impressive successes, yet struggles with sample inefficiency, poor generalization, and limited interpretability. Conversely, large language models (LLMs) excel in reasoning, making them promising for general decision-making and planning. This work presents LLM-Zero, a training-free framework to enhance LLMs’ performance in planning tasks. We evaluate LLM-Zero against leading LLM-based and traditional DRL methods, demonstrating its competitive potential without additional training. These findings highlight LLM-Zero as a promising approach to integrate LLMs into planning and
decision-making tasks.

## Install

#### Prerequisite 

- Linux
- Nvdia GPU with CUDA support

### conda environment

create conda environment with python 3.11 (any versions between 3.9 and 3.11 should work, but only 3.11 is tested).

```bash
conda create -n llmzero python=3.11
conda activate llmzero
```

### Install Dependencies
```bash
pip install -r requirements.txt
pip install -q git+https://github.com/tasbolat1/pyRDDLGym.git --force-reinstall
pip install numpy==1.24.2 --force-reinstall
```

## Reproduce result

We have stored the LLM outputs locally to get reproducible results. Run the script

```sh
sh reproducible.sh
```

## Run with your own API key (not tested)

You can also run on a different seed or add a new environment and run with your own key. To achieve this, create a `.env` file with the following format

```
OPENAI_API_KEY= <your openai api key>
USE_OPENAI_CUSTOM=True
CUSTOM_BASE_URL="https://api.mistral.ai/v1"
CUSTOM_API_KEY= <your mistral api key>
CUSTOM_MODEL_ID="mistral-large-2407"
```
