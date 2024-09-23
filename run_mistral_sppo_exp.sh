#!/bin/bash
i=1
OLD_DATASET_DIR="synthetic_data_reg1_0.01_fixedprompt1-mistral-7b-instruct-sppo-iter1_score"
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"

exp_list=(sppo_forward_importance 0.01 
          sppo_forward_importance 0.001 
          sppo_reversekl 0.1
          sppo_reversekl 0.01)

for ((i=0; i<${#exp_list[@]}; i+=2)); do
    LOSS_TYPE=${exp_list[i]}
    REG_COEF=${exp_list[i+1]}
    OUTPUT_DIR="checkpoints/${LOSS_TYPE}-${REG_COEF}-PromptA-Mistral-7B-Instruct-SPPO-Iter${i}"
    OUT="data-${LOSS_TYPE}-${REG_COEF}-promptA-mistral-7b-instruct-sppo-iter${i}"
    DATASET_DIR="synthetic_data_${LOSS_TYPE}_${REG_COEF}_promptA-mistral-7b-instruct-sppo-iter${i}_score"
    echo "Running experiment with LOSS_TYPE=${LOSS_TYPE}, REG_COEF=${REG_COEF}, OUTPUT_DIR=${OUTPUT_DIR}, OUT=${OUT}, DATASET_DIR=${DATASET_DIR}"
    cp -R $OLD_DATASET_DIR $DATASET_DIR

    # bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
    --dataset $DATASET_DIR \
    --output_dir $OUTPUT_DIR --num 1 --loss_type ${LOSS_TYPE} --reg_coef ${REG_COEF}

done
    



# OUTPUT_DIR="checkpoints/${LOSS_TYPE}-${REG_COEF}-PromptA-Mistral-7B-Instruct-SPPO-Iter${i}"
# PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"
# OUT="data-${LOSS_TYPE}-${REG_COEF}-promptA-mistral-7b-instruct-sppo-iter${i}"
# DATASET_DIR="synthetic_data_${LOSS_TYPE}_${REG_COEF}_promptA-mistral-7b-instruct-sppo-iter${i}_score"
# 
# # bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
# bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
# --dataset $DATASET_DIR \
# --output_dir $OUTPUT_DIR --num 1 --loss_type ${LOSS_TYPE} --reg_coef ${REG_COEF}
