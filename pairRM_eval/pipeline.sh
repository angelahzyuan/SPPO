## Usage: example: bash pipeline.sh iter1_5.0e-7_beta0.01_adamw_torch-sigmoid_baseline_1 0
# bash pipeline.sh iter1_5.0e-7_beta0.01_adamw_torch-sigmoid_baseline_1 1
# bash pipeline.sh iter1_5.0e-7_beta0.01_adamw_torch-sigmoid_baseline_1 2
# bash pipeline.sh iter1_5.0e-7_beta0.01_adamw_torch-sigmoid_baseline_1 3
# Here we use numgpu=32, so we start 4 scripts, with each 8 gpus. How many scripts you can start at once depends on your machine

if [ $# -ne 2 ]; then
    echo "Usage: $0 <name> <num>"
    exit 1
fi

# Assign the input argument to a variable
NAME=$1
num=$2
numgpu=32
PREF="test_"

# Define source and destination directories
SOURCE_DIR="./data/todo"
DEST_DIR="./data"

# Create the destination directory if it does not exist
mkdir -p "$DEST_DIR"

FIGS_DIR="./figs"

# Check for files containing the substring $NAME in $FIGS_DIR
if find "$FIGS_DIR" -type f -name "*$NAME*" | grep -q .; then
    echo "Files containing '$NAME' already exist in $FIGS_DIR. Exiting program."
    exit 0
fi

# Other operations would go here
echo "No files with '$NAME' found in $FIGS_DIR. Continuing operations..."


if compgen -G "$DEST_DIR/${PREF}${NAME}-checkpoint-*" > /dev/null; then
    echo "Files with the pattern ${PREF}${NAME}-checkpoint-* already exist in $DEST_DIR. No files copied."
else
    # No matching files in DEST_DIR, proceed with copy
    echo "No matching files found in $DEST_DIR. Copying files..."
    cp "${SOURCE_DIR}/${PREF}${NAME}-checkpoint-"* "$DEST_DIR"
    echo "Copy operation complete."
fi

RANKING_DIR="./data/ranking"

# Ensure the RANKING_DIR exists, create if not
mkdir -p "$RANKING_DIR"

# Loop from 0 to 7
( for i in {0..7}; do
    # Calculate the GPU index based on the input num
    gpu_index=$((i + num * 8))

    # Define the filename based on the GPU index
    file_name="test_ranks_${gpu_index}.jsonl"

    # Check if the file does not exist in the RANKING_DIR
    if [ ! -f "${RANKING_DIR}/${file_name}" ]; then
        echo "File ${file_name} does not exist, running python script on GPU ${gpu_index}."
        # Run the Python script with the calculated GPU index
        python rank.py --numgpu $numgpu --gpu $gpu_index &
    else
        echo "File ${file_name} already exists, skipping GPU ${gpu_index}."
    fi
done; wait ) & all_rank=$!
wait $all_rank
echo "All operations completed."

# Count the number of files in the ranking directory
file_count=$(find "$RANKING_DIR" -type f | wc -l)

# Compare the file count to numgpu
if [ "$file_count" -eq "$numgpu" ]; then
    echo "Number of files matches numgpu ($numgpu), running summarize.py..."
    python summarize.py --name $NAME

    if [ $? -eq 0 ]; then
        echo "summarize.py completed successfully, proceeding with cleanup."

        # Clean previously copied files (e.g., copied to ./data)
        rm -f "./data/${PREF}${NAME}-checkpoint-"*  # Adjust this pattern to match the files you want to delete
        mv "./data/todo/${PREF}${NAME}-checkpoint-"* "./data/finished/"
        # Clean all files under the ranking directory
        rm -rf "$RANKING_DIR"/*

        echo "Cleanup complete."
    else
        echo "summarize.py failed, skipping cleanup."
    fi
else
    echo "Number of files ($file_count) does not match expected numgpu ($numgpu)."
fi
