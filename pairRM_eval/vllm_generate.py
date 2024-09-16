from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import fire
import json


def generate(
    model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    input_file="data/test_inputs.jsonl",
    prefix: str = "test_",
    save_dir: str = "data/todo",
    temperature: float = 0.7,
    top_p: float = 1.0,
    seq_len: int = 2048,
):
    """
    top_p := vocab frac used until the prob of vocab is top_p
    temp := score/temp so 0 is deterministic.
    """
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, n=1, max_tokens=seq_len
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    prompt_ids = []

    vllm_inputs = []
    with open(input_file, "r") as f:
        for line in f:
            ex = json.loads(line)
            vllm_inputs.append(ex["prompt"])
            prompt_ids.append(ex["prompt_id"])

    def apply_template(text):
        return tokenizer.decode(
            tokenizer.apply_chat_template([{"role": "user", "content": text}]),
            skip_special_tokens=True,
        )

    mistral_vllm_inputs = [apply_template(text) for text in vllm_inputs]

    # llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    # llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", dtype=torch.bfloat16)
    llm = LLM(model=model, dtype=torch.bfloat16)
    # llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16)
    output = llm.generate(mistral_vllm_inputs, sampling_params=sampling_params)

    model_name = "-".join(model.split("/")[-3:])
    output_file_name = f"{save_dir}/{prefix}{model_name}.jsonl"

    # print(output[0].outputs[0].text)
    with open(output_file_name, "w") as f:
        for i, o in enumerate(output):
            f.write(
                json.dumps(
                    {
                        "prompt_id": prompt_ids[i],
                        "prompt": vllm_inputs[i],
                        "generated_response": o.outputs[0].text.strip(),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    fire.Fire(generate)
