import argparse
import torch
import json
import os
import random
import warnings
import numpy as np
import time

import datasets
# from datasets import load_dataset
from vllm import LLM, SamplingParams
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.lora.request import LoRARequest

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

USE_LORA = True
# path_llm = None
# path_llm = "meta-llama/Meta-Llama-3-8B-Instruct"
# path_llm = "mistralai/Mistral-7B-Instruct-v0.2"
# path_llm = "checkpoints/checkpoint-313"
#path_llm = "checkpoints/Llama-3-8B-Instruct-SPPO-LoRA-Iter1"
#path_llm = "checkpoints/Mistral-7B-Instruct-SPPO-Iter3"
#path_llm = "checkpoints/Mis7B-It-SPPO-LoRA128-Iter1"
#path_llm = "checkpoints/Mis7B-It-SPPO-LoRA64-Iter1"
path_llm = "checkpoints/Mistral-7B-It-SPPO-LoRA8-Iter3"
name_file = "mistral-lora8-iter3"

if USE_LORA:
    with open(path_llm + "/adapter_config.json", 'r') as json_data:
        config_adapter = json.load(json_data)
        print(config_adapter)
    
    path_basemodel = config_adapter['base_model_name_or_path']
    llm = LLM(
      model=path_basemodel, 
      tensor_parallel_size=1,
      enable_lora=True,
      max_lora_rank=config_adapter["r"]
    )
else:
    llm = LLM(
      model=path_llm, 
      tensor_parallel_size=1,
    )

tokenizer = AutoTokenizer.from_pretrained(path_llm)
tokenizer.pad_token = tokenizer.eos_token

# Load evaluation dataset for AlpacaEval
eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

# Generate evaluation responses from the model
res = list()
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    seed=2024,
    max_tokens=2048,
    #max_tokens=64, # set it to higher value like 2048 for proper test
)

time_start = time.time()

max_examples = -1 # Set this to a positive value to test on smaller number of prompts
idx_examples = 0
for example in eval_set:
    ''' Tested with
        - Meta-Llama-3-8B
        - Mistral-7B-Instruct-v0.2
    '''
    example['prompt_template'] = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["instruction"]}, 
            {"role": "assistant", "content": "None"}
        ],
        tokenize=False, add_generate_prompt=True
    ).split("None")[0]

    if USE_LORA:
        example["output"] = llm.generate(
            # example["instruction"],
            example["prompt_template"],
            sampling_params,
            lora_request=LoRARequest("sql_adapter", 1, path_llm)
        )[0].outputs[0].text
    else:
        example["output"] = llm.generate(
            # example["instruction"],
            example["prompt_template"],
            sampling_params
        )[0].outputs[0].text
        
    example["generator"] = name_file # name of your model
    res.append(example)

    idx_examples += 1
    if max_examples > 0 and idx_examples == max_examples:
        break

time_elapsed = time.time() - time_start
print(f"{time_elapsed / 3600:.2f} hours passed to finish generating {idx_examples} responses")

# Save generated responses
path_savedir = "./results_alpacaeval/"
os.makedirs(path_savedir, exist_ok=True)
with open(f"{path_savedir}/{name_file}.json", "w") as f:
    json.dump(res, f)

# Set OpenAI API key with 'export OPENAI_API_KEY=YOUR_KEY'.
# Run $ alpaca_eval --model_outputs 'test_alpacaeval/responses.json'.
    
