#!/bin/bash
iter_num=1
for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    else
        MODEL="checkpoints/FixedPrompt1-Mistral-7B-Instruct-SPPO-Iter$((i-1))"
    fi
    OUTPUT_DIR="checkpoints/Entropy0.1-FixedPrompt1-Mistral-7B-Instruct-SPPO-Iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"
    OUT="data-entropy0.1-fixedprompt1-mistral-7b-instruct-sppo-iter${i}"

    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
    --dataset "synthetic_data_entropy0.1_fixedprompt1-mistral-7b-instruct-sppo-iter${i}_score" \
    --output_dir $OUTPUT_DIR --num 1 --loss_type "sppo_entropy" --reg_coef 0.1
done
