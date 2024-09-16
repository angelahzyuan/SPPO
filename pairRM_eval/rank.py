from datasets import load_dataset, Dataset
import json
from tqdm import tqdm

import pandas as pd
import argparse
import llm_blender
import os
import numpy as np
from copy import deepcopy
import glob


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--test_inputs_file",
        type=str,
        default="data/test_inputs.jsonl",
    )
    parser.add_argument(
        "--generated_pattern",
        type=str,
        default="test_",
    )
    parser.add_argument("--numgpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)  # local rank

    return parser.parse_args()


def ranking(prompts, candidates, gpu, outdir):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM", device=f"cuda:{gpu}")  # load PairRM
    ranks = blender.rank(prompts, candidates, return_scores=True, batch_size=1)
    np.save(f"ranking/{outdir}/{gpu}.npy", ranks)


def main(args):
    # Load data

    test_inputs_file = args.test_inputs_file

    # test_inputs_file = "/home/yikangs/zhiqings/sppo/Snorkel-Eval/data/test_inputs.jsonl"

    data = []
    with open(test_inputs_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # sort data by prompt_id
    data = sorted(data, key=lambda x: x["prompt_id"])

    prompts_all = [data[idx]["prompt"] for idx in range(len(data))]

    all_generated = []

    generated_pattern = (
        f"data/{args.generated_pattern}*.jsonl"
    )

    model_names = []

    for generated_file in glob.glob(generated_pattern):
        if generated_file != test_inputs_file:
            with open(generated_file, "r") as f:
                generated = []
                for line in f:
                    generated.append(json.loads(line))
            generated = sorted(generated, key=lambda x: x["prompt_id"])
            generated = [g["generated_response"] for g in generated]
            all_generated.append(generated)

            model_names.append(generated_file.split("/")[-1].split(".json")[0])

    all_generated = list(zip(*all_generated))

    # prompts_all = pr
    # ompts_all[:16]
    # all_generated = all_generated[:16]

    blender = llm_blender.Blender()
    gpu = args.gpu
    blender.loadranker("llm-blender/PairRM", device=f"cuda:{gpu%8}")  # load PairRM
    prompts_all = [p for i, p in enumerate(prompts_all) if i % args.numgpu == gpu]
    all_generated = [c for i, c in enumerate(all_generated) if i % args.numgpu == gpu]
    ranks = blender.rank(
        prompts_all, all_generated, return_scores=True, batch_size=1, policy="raw"
    )
    os.makedirs(f"{args.output_dir}/ranking", exist_ok=True)

    # merge ranks with prompts_all & generated

    with open(
        f"{args.output_dir}/ranking/{args.generated_pattern}ranks_{gpu}.jsonl", "w"
    ) as f:
        for i in range(len(prompts_all)):
            f.write(
                json.dumps(
                    {
                        "prompt": prompts_all[i],
                        "ranks": ranks[i].tolist(),
                        "model_names": model_names,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
