import argparse
import torch
import json
import os
import random
import warnings
import numpy as np
import pandas as pd
import time

import datasets
import os
import nltk

from multiprocessing import Pool
from vllm import LLM, SamplingParams
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.lora.request import LoRARequest
from nltk.translate.bleu_score import SmoothingFunction
from abc import abstractmethod

nltk.download("punkt_tab")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# path_llm = None
# path_llm = "meta-llama/Meta-Llama-3-8B-Instruct"
# path_llm = "mistralai/Mistral-7B-Instruct-v0.2"
# path_llm = "checkpoints/checkpoint-313"
#path_llm = "checkpoints/Llama-3-8B-Instruct-SPPO-LoRA-Iter1"
#path_llm = "checkpoints/Mistral-7B-Instruct-SPPO-Iter3"
#path_llm = "checkpoints/Mis7B-It-SPPO-LoRA128-Iter1"
#path_llm = "checkpoints/Mis7B-It-SPPO-LoRA64-Iter1"
# path_llm = "checkpoints/Mistral-7B-It-SPPO-LoRA8-Iter3"
path_llm = "google/gemma-2b-it"

USE_LORA = False
name_file = "test0-selfbleu-gemma2bit"
num_samples = 4 # number of responses per prompt for estimating diversity
max_examples = -1 # Set this to a positive value to test on smaller number of prompts

class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

class SelfBleu(Metrics):
    def __init__(self, test_text:list=[], gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            # with open(self.test_data) as real_data:
                # for text in real_data:
                #     text = nltk.word_tokenize(text)
                #     reference.append(text)
            for text in self.test_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
                
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        for hypothesis in self.test_data:
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

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
eval_set = datasets.load_dataset(
    "tatsu-lab/alpaca_eval", 
    "alpaca_eval", 
    trust_remote_code=True
)["eval"]

# Generate evaluation responses from the model
cols = ["model", "prompt", "score_selfBLEU"] + ["response" + str(i+1) for i in range(num_samples)]
res = list()
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    seed=2024,
    max_tokens=2048, # set it to higher value like 2048 for proper test
    n=num_samples
)

time_start = time.time()

idx_examples = 0
for example in eval_set:
    ''' Tested with
        - Meta-Llama-3-8B
        - Mistral-7B-Instruct-v0.2
    '''
    prompt_template = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["instruction"]}, 
            {"role": "assistant", "content": "None"}
        ],
        tokenize=False, add_generate_prompt=True
    ).split("None")[0]

    if USE_LORA:
        outputs = llm.generate(
            prompt_template,
            sampling_params,
            lora_request=LoRARequest("sql_adapter", 1, path_llm)
        )
    else:
        outputs = llm.generate(
            prompt_template,
            sampling_params
        )

    outputs = [
        output1.text for output0 in outputs for output1 in output0.outputs
    ]

    # evaluate self-BLEU
    sb = SelfBleu(outputs).get_score()
    
    res.append([path_llm, prompt_template, sb] + outputs)

    idx_examples += 1
    if max_examples > 0 and idx_examples == max_examples:
        break

time_elapsed = time.time() - time_start
print(f"{time_elapsed / 3600:.2f} hours passed to finish generating {idx_examples} responses ({num_samples} responses per prompt)")

# Save generated responses
path_savedir = "./results_selfbleu/"
os.makedirs(path_savedir, exist_ok=True)
df = pd.DataFrame(res, columns=cols)
df.to_csv(path_savedir + "results_" + name_file + ".csv", index=False)
mean_sb = df["score_selfBLEU"].mean()

with open(path_savedir + "summary_" + name_file + ".txt", "w") as fp:
    fp.write(f"""
        used model: {path_llm}
        USE_LORA: {USE_LORA}
        name_file: {name_file}
        num_samples: {num_samples}
        max_examples: {idx_examples}
        
        =====
        {time_elapsed / 3600:.2f} hours passed to finish generating {idx_examples} responses ({num_samples} responses per prompt)
        mean self-BLEU score: {mean_sb} (lower score means more diverse responses)
    """)


    
