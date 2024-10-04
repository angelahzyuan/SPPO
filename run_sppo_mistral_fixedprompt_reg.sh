#!/bin/bash
iter_num=6
LOSS_TYPE=sppo_forward_importance
# LOSS_TYPE=sppo_reversekl
# LOSS_TYPE=sppo
REG_COEF=0.1
# REG_COEF=0

for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    else
        MODEL="checkpoints/${LOSS_TYPE}-${REG_COEF}-PromptA-Mistral-7B-Instruct-SPPO-Iter$((i-1))"
    fi
    OUTPUT_DIR="checkpoints/${LOSS_TYPE}-${REG_COEF}-PromptA-Mistral-7B-Instruct-SPPO-Iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"
    OUT="data-${LOSS_TYPE}-${REG_COEF}-promptA-mistral-7b-instruct-sppo-iter${i}"
    DATASET_DIR="synthetic_data_${LOSS_TYPE}-${REG_COEF}-promptA-mistral-7b-instruct-sppo-iter${i}_score"

    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
    --dataset $DATASET_DIR \
    --output_dir $OUTPUT_DIR --num 1 --loss_type ${LOSS_TYPE} --reg_coef ${REG_COEF}
done
