Usage example

- bash generate.sh ../checkpoints/Mistral-7B-Instruct-SPPO-Iter1 309
- bash pipeline.sh Mistral-7B-Instruct-SPPO-Iter1 0
- bash pipeline.sh Mistral-7B-Instruct-SPPO-Iter1 1
- bash pipeline.sh Mistral-7B-Instruct-SPPO-Iter1 2
- bash pipeline.sh Mistral-7B-Instruct-SPPO-Iter1 3

Might have to check the generated results folder for accurate $NAME, you will see a file named `generated/test_{some_name}_checkpoint-{some_number}`, the name zone here should be ${some_name}

push_model_to_hub.py is for pushing trained model to huggingface hub.
