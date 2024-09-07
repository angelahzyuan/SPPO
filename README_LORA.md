
# Installation

The procedure below is recommended to avoid python library compatibility issue.

```
$ pip install vllm==0.5.0.post1 # To maintain compatibility with SPPO
$ git clone https://github.com/yuchenlin/LLM-Blender.git
$ cd LLM-Blender
$ pip install -e .
$ cd ..
$ cd self-alignment
$ pip install -e .
$ huggingface-cli login # Make sure you are using the key with 'write' permissions.
```

## Configuration Advices (20th Aug 2024, Seongho)
- Configure huggingface organization(id) at `run_sppo_lora*.sh` ('HF_ORG') 
- Configure GPU allocation settings in 
	- `scripts/generate.sh` ('AVAILABLE_GPUS')
	- `scripts/pipeline.sh` ('OMP_NUM_THREADS', 'CUDA_AVAILABLE_DEVICES')
	- `recipes/accelerate_configs/*.yaml` ('num_processes')
- If you want to reduce the size of the training data, set `SIZE_TRAIN` in `run_sppo_MODEL.sh` to a positive integer (ex. 32). 

# Local Changes 

## Data Generation
- Specify your available GPU ids and huggingface id in `scripts/generate.sh`. GPU ids don't have to start from 0 or be consecutive.

## Pipeline
- Currently only manage to run PEFT on single card of NVIDIA Corporation GA100 [A100 SXM4 80GB].

## Using LoRA
- Set `USE_LORA="true"` in your run_sppo script. Refer to `run_sppo_lora*.sh` for example. 
- **(4th September 2024) IMPORTANT: `vllm` supports `max_lora_rank` only up to 64, so setting rank higher than this will make the code abort after the first iteration of training.**
- Set `zero3_init_flag: false` in `recipes/accelerate_configs/deepspeed_zero3.yaml`.
	- **Not tested**: This might affect using multi-gpu environment with LoRA. 

# Using AlpacaEval

## Installing a Separate Virtual Environment
- Use python version higher than or equal to 3.10 (does not run on 3.9)
- This is required because SPPO repository depends on older versions of some libraries, for example `vllm`.
- Create a separate virtual environment, and `pip install alpaca_eval`, `pip install vllm`.
- Note the path to `alpaca_eval` inside the virtual environment, for example `venv_alpacaeval/lib/python3.10/site-packages/alpaca_eval`. This path will be denoted as `PATH_AEVAL` below.
- Conda: `PATH_AEVAL=miniconda3/envs/rlhf_eval/lib/python3.10/site-packages/alpaca_eval`
 
## Creating a Directory for Submission 
- inside `PATH_AEVAL/models_configs/`, create a directory for the submission to be stored. The name of this directory will be denoted as `NAME_SUBMISSION` below.
- Create `configs.yaml` and `prompt.txt` inside the directory.
	- Refer to `PATH_AEVAL/models_configs/SPPO-Mistral7B-PairRM` for example.
	- (21st Aug 2024) `prompt.txt` might need to have `<s>` in the beginning. 
- If `configs.yaml` is copied from another directory, properly modify it.
	- The first line should match the name of the directory.
	- `completions_kwargs/model_name`: It can be either a path in huggingface or local directory where the model is stored. Give the path to the directory where `.safetensors` files are stored.
	- `pretty_name`: this name will be posted on the leaderboard. If this overlaps with one of the names existing on the leaderboard, the evaluation will not run (and show the old leaderboard).
	- `link`: According to Huizhuo, this link does not have to be a correct one.

## Running Evaluation
- Set OpenAI API key with `$ export OPENAI_API_KEY=YOUR_KEY`.
- (With virtual environment for alpaca_eval activated) Run `$ alpaca_eval evaluate_from_model NAME_SUBMISSION`.
- These will happen:
	- The model will be loaded or downloaded from huggingface to your machine.
	- The model will be run on the provided evaluation prompts to generate responses.
	- This response will be used with baseline reponses for comparison, which is done by using OpenAI API.
	- Once this is done, the statistics will be provided with your submission being part of the leaderboard.

## Manual Generation of Responses Using `test_alpacaeval.py`
- (27th Aug. 2024) This file supports LoRA evaluation using `vllm`, while current `alpaca_eval` does not.
- Parameters to configure
	- USE_LORA
	- path_llm: path where the checkpoint is stored
	- name_file: name of `.json` file to be created
	- parameters in `sampling_params` if necessary
- Use the same virtual environment for running `alpaca_eval`. After configuring parameters, run `$ python3 test_alpacaeval.py`.
- After the response is generated, run `alpaca_eval --model_outputs 'results_alpacaeval/NAME_FILE.json'` to evaluate the responses.
	- Set OpenAI API key with `export OPENAI_API_KEY=YOUR_KEY`.
