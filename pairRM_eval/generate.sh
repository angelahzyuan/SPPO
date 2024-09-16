### Usage: example: bash generate.sh ../checkpoints/Mistral-7B-Instruct-SPPO-Iter1 309
### The original script are used to generate multiple epochs at a time, for exampel checkpoint-309, checkpoint-618, etc. If we only have the checkpoint for one epoch, only 1 gpu would be utilized

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_directory> <num>"
    exit 1
fi

# Assign the input arguments to variables
MODEL_DIR=$1
NUM=$2

( for gpu_id in {0..7}; do
    # Calculate the checkpoint number
    checkpoint=$((NUM * (gpu_id + 1)))
    model_path="${MODEL_DIR}/checkpoint-${checkpoint}"
    CUDA_VISIBLE_DEVICES=$gpu_id python vllm_generate.py --model "$model_path" &
done; wait ) & all_gen=$!

wait $all_gen

# ( for gpu_id in {0..7}; do
#     CUDA_VISIBLE_DEVICES=$gpu_id python3 vllm_generate.py --model $MODEL
# done; wait ) & all_gen=$!

# wait $all_gen
