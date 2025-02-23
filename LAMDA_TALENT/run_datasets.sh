#!/bin/bash

# Function to check dataset size from info.json
check_and_run() {
    local dataset_path=$1
    local dataset_name=$(basename "${dataset_path}")
    
    echo "Debug: Processing ${dataset_path}"
    
    # Check if RFM results already exist
    if [ -f "../rfm_results/${dataset_name}-rfm.pkl" ]; then
        echo "Skipping ${dataset_name} - RFM results already exist"
        skipped_datasets+=("${dataset_name} (results exist)")
        ((skipped_existing++))
        return 0
    fi
    
    if [ ! -f "${dataset_path}/info.json" ]; then
        echo "Error: info.json not found in ${dataset_path}"
        return 1
    fi
    
    train_size=$(grep -o '"train_size": [0-9]*' "${dataset_path}/info.json" | cut -d' ' -f2)
    if [ -z "$train_size" ]; then
        echo "Error: Could not find train_size in info.json for ${dataset_path}"
        return 1
    fi
    
    echo "Dataset: ${dataset_path}, Train size: ${train_size}"
    
    if [ "$train_size" -lt 50000 ]; then
        echo "Attempting to launch job for ${dataset_name}"
        python_cmd="python train_model_classical.py --dataset $dataset_name"
        echo "Command to execute: ${python_cmd}"
        if sbatch --job-name="${dataset_name}" delta_setup "${python_cmd}"; then
            echo "Successfully submitted job for ${dataset_name}"
        else
            echo "Error: Failed to submit job for ${dataset_name}"
            return 1
        fi
    else
        echo "Dataset ${dataset_path} is too large (size: ${train_size})"
        skipped_datasets+=("${dataset_name} (too large: ${train_size})")
        ((skipped_large++))
    fi
}

# Main script
count=0
# Initialize array and counters
declare -a skipped_datasets
skipped_existing=0
skipped_large=0
max_datasets=300

echo "Debug: Starting script"
echo "Debug: Looking in ../datasets/ directory"

# Check if parent datasets directory exists
if [ ! -d "../datasets" ]; then
    echo "Error: datasets directory not found in parent directory"
    exit 1
fi

# Changed the path to look in parent directory
for dataset in ../datasets/*/; do
    if [ -d "$dataset" ] && [ $count -lt $max_datasets ]; then
        echo "Debug: Processing dataset #$((count+1)): $dataset"
        check_and_run "$dataset"
        ((count++))
    fi
done

echo "Debug: Script completed. Processed $count datasets"
echo -e "\nSkip Summary:"
echo "Total datasets skipped: $((skipped_existing + skipped_large))"
echo "- Skipped (existing results): $skipped_existing"
echo "- Skipped (too large): $skipped_large"
echo -e "\nDetailed list of skipped datasets:"
if [ ${#skipped_datasets[@]} -eq 0 ]; then
    echo "No datasets were skipped"
else
    printf '%s\n' "${skipped_datasets[@]}"
fi