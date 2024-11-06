import os
import json
import random

DATA_DIR = "data"

from datasets import load_dataset

dataset = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")

all_prompt_ids = []

id_to_prompt = {}

for ex in dataset["test_iteration_3"]:
    all_prompt_ids.append(ex["prompt_id"])
    id_to_prompt[ex["prompt_id"]] = ex["prompt"]

with open(os.path.join(DATA_DIR, "test_inputs.jsonl"), "w") as f:
    for ex in dataset["test_iteration_3"]:
        f.write(
            json.dumps(
                {
                    "prompt_id": ex["prompt_id"],
                    "prompt": ex["prompt"],
                }
            )
            + "\n"
        )

for data_name in ["test_iteration_1", "test_iteration_2", "test_iteration_3"]:
    all_prompt_ids_set = set(all_prompt_ids)

    with open(os.path.join(DATA_DIR, f"{data_name}.jsonl"), "w") as f:
        for ex in dataset[data_name]:
            assert ex["prompt_id"] in all_prompt_ids_set

            num_generated_responses = len(ex["all_generated_responses"])
            response_id = random.randint(0, num_generated_responses - 1)

            f.write(
                json.dumps(
                    {
                        "prompt_id": ex["prompt_id"],
                        "prompt": ex["prompt"],
                        "generated_response": ex["all_generated_responses"][
                            response_id
                        ],
                    }
                )
                + "\n"
            )
            all_prompt_ids_set.remove(ex["prompt_id"])

        print("Remaining prompt ids:", all_prompt_ids_set)

        for prompt_id in all_prompt_ids_set:
            f.write(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "prompt": id_to_prompt[prompt_id],
                        "generated_response": "",
                    }
                )
                + "\n"
            )

for data_name in ["test_iteration_1", "test_iteration_2", "test_iteration_3"]:
    all_prompt_ids_set = set(all_prompt_ids)

    with open(os.path.join(DATA_DIR, f"{data_name}_best_of_5.jsonl"), "w") as f:
        for ex in dataset[data_name]:
            assert ex["prompt_id"] in all_prompt_ids_set

            all_scores = ex["all_rm_scores"]
            best_response_id = all_scores.index(max(all_scores))

            f.write(
                json.dumps(
                    {
                        "prompt_id": ex["prompt_id"],
                        "prompt": ex["prompt"],
                        "generated_response": ex["all_generated_responses"][
                            best_response_id
                        ],
                    }
                )
                + "\n"
            )
            all_prompt_ids_set.remove(ex["prompt_id"])

        print("Remaining prompt ids:", all_prompt_ids_set)

        for prompt_id in all_prompt_ids_set:
            f.write(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "prompt": id_to_prompt[prompt_id],
                        "generated_response": "",
                    }
                )
                + "\n"
            )
