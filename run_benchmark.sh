#!/bin/bash
# run_benchmarks.sh - Script to automate GPU benchmarking
# 
# This script runs both the Stable Diffusion and LLM benchmarks
# with appropriate parameters for different GPUs

# Configuration
OUTPUT_DIR="benchmark_results"
SD_MODEL="stabilityai/stable-diffusion-2-1"
LLM_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"  # Change as needed

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Detect GPU information
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '[:space:]' | tr -d ',')
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    echo "Detected GPU: $GPU_NAME with $GPU_MEMORY"
else
    echo "No NVIDIA GPU detected. Please specify GPU name manually."
    read -p "Enter GPU name: " GPU_NAME
fi

# Determine benchmark parameters based on GPU memory
if [[ "$GPU_MEMORY" == *"24"* ]] || [[ "$GPU_MEMORY" == *"32"* ]] || [[ "$GPU_MEMORY" == *"40"* ]] || [[ "$GPU_MEMORY" == *"48"* ]] || [[ "$GPU_MEMORY" == *"80"* ]]; then
    # High memory GPUs (24GB+)
    RESOLUTIONS="512x512,768x768,1024x1024,1536x1536"
    STEPS="20,50"
    SEQ_LENGTHS="128,512,1024,2048"
    BATCH_SIZES="1,4,8"
    echo "Using parameters for high-memory GPU"
elif [[ "$GPU_MEMORY" == *"16"* ]] || [[ "$GPU_MEMORY" == *"20"* ]]; then
    # Mid-range GPUs (16-20GB)
    RESOLUTIONS="512x512,768x768,1024x1024"
    STEPS="20,50"
    SEQ_LENGTHS="128,512,1024"
    BATCH_SIZES="1,4"
    echo "Using parameters for mid-range GPU"
else
    # Lower memory GPUs (<16GB)
    RESOLUTIONS="512x512,768x768"
    STEPS="20,50"
    SEQ_LENGTHS="128,512"
    BATCH_SIZES="1,2"
    echo "Using parameters for lower-memory GPU"
fi

# Run the benchmark with appropriate parameters
echo "Starting benchmarks at $(date)"
echo "==============================================="

# Now run the benchmark with the correct command line arguments
echo "Running benchmark for $GPU_NAME..."
python gpu_benchmark.py \
    --gpu "$GPU_NAME" \
    --sd-model "$SD_MODEL" \
    --llm-model "$LLM_MODEL" \
    --resolutions "$RESOLUTIONS" \
    --steps "$STEPS" \
    --seq-lengths "$SEQ_LENGTHS" \
    --batch-sizes "$BATCH_SIZES" \
    --samples 3 \
    --output-dir "$OUTPUT_DIR"

echo "==============================================="
echo "Benchmarks completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
