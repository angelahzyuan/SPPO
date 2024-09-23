#!/bin/bash
iter_num=1
for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    else
        MODEL="checkpoints/FixedPrompt1-Mistral-7B-Instruct-SPPO-Iter$((i-1))"
    fi
    OUTPUT_DIR="checkpoints/ForwardImportance0.01-FixedPrompt1-Mistral-7B-Instruct-SPPO-Iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"
    OUT="data-forwardimportance0.01-fixedprompt1-mistral-7b-instruct-sppo-iter${i}"

    # bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
    --dataset "synthetic_data_forwardimportance0.01_fixedprompt1-mistral-7b-instruct-sppo-iter${i}_score" \
    --output_dir $OUTPUT_DIR --num 1 --loss_type "sppo_forward_importance" --reg_coef 0.01
done
