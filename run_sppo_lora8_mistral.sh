#!/bin/bash
iter_num=1
LORA_R=8 # size of LoRA rank - one of 8, 16, 32, 64
USE_LORA="true" # "true" or "false"
SIZE_TRAIN=-1 # set to -1 for default (full data)
#SIZE_TRAIN=7 # set to -1 for default (full data)
HF_ORG="geronest"
PAIRS=5
# PAIRS=1
GPU=2
TRAINING_BATCH_SIZE=1

for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Mistral-7B-It-SPPO-LoRA${LORA_R}-Iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"
    OUT="data-mis7b-it-sppo-lora${LORA_R}-iter${i}"

    bash scripts/generate.sh --gpu $GPU --model $MODEL --prompt $PROMPT --out_path $OUT --iter $i --use_lora $USE_LORA --size_train $SIZE_TRAIN --pairs $PAIRS --hf_org $HF_ORG
    bash scripts/pipeline.sh --gpu $GPU --batch_size $TRAINING_BATCH_SIZE --model $MODEL --iter $i --dataset "synthetic_data_mis7b-it-sppo-lora${LORA_R}-iter${i}_score" --output_dir $OUTPUT_DIR --num 1 --lora_r $LORA_R --use_lora $USE_LORA --size_train $SIZE_TRAIN
done
