#!/usr/bin/env python3
"""
GPU Benchmarking Tool

This script benchmarks GPU performance for both:
1. Stable Diffusion image generation at specified resolutions and step counts
2. LLM inference speed in tokens per second

Usage:
    python gpu_benchmark.py --gpu [gpu_name] [other options]
"""

import argparse
import json
import os
import time
import platform
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# For Stable Diffusion
try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    print("Warning: diffusers package not found. Stable Diffusion benchmarks will not be available.")

# For LLM Benchmarks
try:
    import transformers
except ImportError:
    print("Warning: transformers package not found. LLM benchmarks will not be available.")

class GPUBenchmark:
    def __init__(self, gpu_name=None, output_dir="benchmark_results"):
        """Initialize the benchmark tool with GPU info and setup output directory."""
        self.gpu_name = gpu_name or self._detect_gpu()
        self.output_dir = output_dir
        self.results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self._get_system_info(),
            "sd_benchmarks": [],
            "llm_benchmarks": []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _detect_gpu(self):
        """Detect the GPU being used."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        else:
            return "CPU (No GPU detected)"
    
    def _get_system_info(self):
        """Gather system information."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A",
            "gpu": self.gpu_name,
            "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if torch.cuda.is_available() else 0
        }
        return info

    def benchmark_stable_diffusion(self, model_id="stabilityai/stable-diffusion-2-1", 
                                 resolutions=[(512, 512), (768, 768), (1024, 1024)], 
                                 step_counts=[20, 50], 
                                 num_samples=3,
                                 prompt="A beautiful landscape with mountains and a lake"):
        """
        Benchmark Stable Diffusion image generation at various resolutions and step counts.
        
        Args:
            model_id: The model ID to load from HuggingFace
            resolutions: List of (width, height) tuples to test
            step_counts: List of step counts to test
            num_samples: Number of samples to generate and average results
            prompt: The prompt to use for generation
        """
        print(f"\n{'='*60}\nBenchmarking Stable Diffusion: {model_id}\n{'='*60}")
        
        try:
            # Print CUDA diagnostics
            print("\nCUDA Diagnostics:")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                print(f"Current CUDA Device: {torch.cuda.current_device()}")
                print(f"Device Name: {torch.cuda.get_device_name()}")
                print(f"Device Capability: {torch.cuda.get_device_capability()}")
                print(f"Device Properties: {torch.cuda.get_device_properties(0)}")
            
            # Try to load with safety checkers disabled and more compatibility options
            print("\nLoading model with compatibility options...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                safety_checker=None,                 # Disable safety checker for performance
                requires_safety_checker=False,       # Avoid safety checker overhead
                variant="fp16",                      # Try fp16 variant if available
                local_files_only=False               # Allow downloading if needed
            )
            
            # Move to CPU first and then to CUDA with specific options
            pipe = pipe.to("cpu")
            
            # Try to enable sliced attention if available
            if hasattr(pipe, "enable_attention_slicing"):
                print("Enabling attention slicing for better compatibility...")
                pipe.enable_attention_slicing(slice_size="auto")
            
            # Try to enable model offloading if memory is tight
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                print("Enabling sequential CPU offload for memory optimization...")
                pipe.enable_sequential_cpu_offload()
            
            # Try to enable xformers if available
            try:
                if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                    print("Enabling xformers for memory efficiency...")
                    pipe.enable_xformers_memory_efficient_attention()
            except Exception as xformer_err:
                print(f"Could not enable xformers: {xformer_err}")
            
            # Now move to CUDA
            print("Moving model to CUDA...")
            pipe = pipe.to("cuda")
            
            success_count = 0
            error_count = 0
            
            # Run benchmarks for each configuration
            for width, height in resolutions:
                for num_steps in step_counts:
                    print(f"\nTesting resolution {width}x{height} with {num_steps} steps")
                    
                    try:
                        times = []
                        for i in range(num_samples):
                            # Clear CUDA cache
                            torch.cuda.empty_cache()
                            
                            # Warm-up run (not measured)
                            if i == 0:
                                print("Running warm-up generation...")
                                generator = torch.Generator("cuda").manual_seed(42)  # Fixed seed for reproducibility
                                _ = pipe(
                                    prompt, 
                                    num_inference_steps=num_steps, 
                                    height=height, 
                                    width=width,
                                    generator=generator
                                )
                                torch.cuda.empty_cache()
                                print("Warm-up complete, starting timed runs")
                            
                            # Timed run
                            start_time = time.time()
                            generator = torch.Generator("cuda").manual_seed(42 + i)  # Different seeds for each sample
                            _ = pipe(
                                prompt, 
                                num_inference_steps=num_steps, 
                                height=height, 
                                width=width,
                                generator=generator
                            )
                            torch.cuda.synchronize()  # Wait for GPU to finish
                            end_time = time.time()
                            
                            times.append(end_time - start_time)
                            print(f"  Sample {i+1}/{num_samples}: {times[-1]:.3f} seconds")
                        
                        # Calculate stats
                        avg_time = np.mean(times)
                        std_dev = np.std(times)
                        
                        # Store results
                        self.results["sd_benchmarks"].append({
                            "resolution": f"{width}x{height}",
                            "steps": num_steps,
                            "avg_time_seconds": avg_time,
                            "std_dev_seconds": std_dev,
                            "samples": num_samples,
                            "images_per_second": 1.0 / avg_time
                        })
                        
                        print(f"  Average: {avg_time:.3f} seconds (±{std_dev:.3f})")
                        print(f"  Performance: {1.0/avg_time:.3f} images/second")
                        
                        success_count += 1
                        
                    except Exception as e:
                        print(f"  Error at resolution {width}x{height}, steps {num_steps}: {e}")
                        error_count += 1
                        # Continue with next configuration rather than failing completely
                        continue
            
            print(f"\nSD Benchmark Summary: {success_count} configurations succeeded, {error_count} failed")
            
            return success_count > 0
        
        except Exception as e:
            print(f"Error benchmarking Stable Diffusion: {e}")
            print("\nTrying to provide more diagnostic information:")
            import traceback
            traceback.print_exc()
            return False

    def benchmark_llm(self, model_id="meta-llama/Llama-2-7b-hf", 
                    sequence_lengths=[128, 512, 1024], 
                    batch_sizes=[1, 4], 
                    num_samples=3,
                    test_type="generation"):
        """
        Benchmark LLM inference speed.
        
        Args:
            model_id: The model ID to load from HuggingFace
            sequence_lengths: List of sequence lengths to test
            batch_sizes: List of batch sizes to test
            num_samples: Number of runs to average
            test_type: Either "generation" or "inference"
        """
        print(f"\n{'='*60}\nBenchmarking LLM: {model_id}\n{'='*60}")
        
        try:
            # Load tokenizer and model with error handling
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("\nLoading tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            except Exception as e:
                print(f"Error loading fast tokenizer, falling back to slow tokenizer: {e}")
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            
            print("\nLoading model with optimized settings...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True  # Required for some models like Qwen
                )
            except Exception as e:
                print(f"Error with optimized loading, trying with basic settings: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to("cuda")
            
            # Try to enable memory optimizations if available
            try:
                print("\nAttempting to apply memory optimizations...")
                if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
                    if "flash_attention_2" in str(model.config.attn_implementation):
                        print("Flash Attention 2 is already enabled")
                    else:
                        print("Attempting to enable Flash Attention...")
                        # This is a config change attempt that might work for some models
                        model.config.attn_implementation = "flash_attention_2"
            except Exception as optimize_err:
                print(f"Optimization attempt error (non-critical): {optimize_err}")
            
            # Use a consistent input text
            base_text = """The history of artificial intelligence dates back to ancient times, when 
            philosophers and mathematicians contemplated the possibility of artificial beings and 
            mechanical men capable of reasoning. However, AI as a formal academic discipline began 
            in the mid-20th century, with the seminal Dartmouth Conference of 1956 often considered 
            the birth of the field. Since then, AI has experienced periods of optimism followed by 
            disappointment and funding cuts, known as "AI winters." Today, machine learning, especially 
            deep learning, has transformed the field, enabling remarkable advances in computer vision, 
            natural language processing, robotics, and numerous other applications that impact our daily lives."""
            
            success_count = 0
            error_count = 0
            
            # Run benchmarks for each configuration
            for seq_length in sequence_lengths:
                for batch_size in batch_sizes:
                    print(f"\nTesting sequence length {seq_length} with batch size {batch_size}")
                    
                    try:
                        # Prepare input
                        input_text = base_text * (seq_length // len(base_text.split()) + 1)
                        input_text = ' '.join(input_text.split()[:seq_length])
                        inputs = tokenizer(
                            [input_text] * batch_size, 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=seq_length
                        ).to("cuda")
                        
                        times = []
                        for i in range(num_samples):
                            # Clear CUDA cache
                            torch.cuda.empty_cache()
                            
                            # Warm-up run (not measured)
                            if i == 0:
                                print("Running warm-up generation...")
                                with torch.no_grad():
                                    _ = model.generate(
                                        **inputs, 
                                        max_new_tokens=20,  # Use fewer tokens for warm-up
                                        do_sample=False
                                    )
                                torch.cuda.empty_cache()
                                print("Warm-up complete, starting timed runs")
                            
                            # Timed run
                            print(f"  Running sample {i+1}/{num_samples}...")
                            start_time = time.time()
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs, 
                                    max_new_tokens=50,
                                    do_sample=False
                                )
                            torch.cuda.synchronize()  # Wait for GPU to finish
                            end_time = time.time()
                            
                            # Calculate tokens per second (input + output tokens)
                            input_tokens = inputs.input_ids.shape[1] * batch_size
                            output_tokens = outputs.shape[1] * batch_size - input_tokens
                            total_tokens = input_tokens + output_tokens
                            tokens_per_second = total_tokens / (end_time - start_time)
                            
                            times.append(end_time - start_time)
                            print(f"  Sample {i+1}/{num_samples}: {times[-1]:.3f} seconds, {tokens_per_second:.1f} tokens/sec")
                        
                        # Calculate stats
                        avg_time = np.mean(times)
                        std_dev = np.std(times)
                        avg_tokens_per_second = total_tokens / avg_time
                        
                        # Store results
                        self.results["llm_benchmarks"].append({
                            "model_id": model_id,
                            "sequence_length": seq_length,
                            "batch_size": batch_size,
                            "avg_time_seconds": avg_time,
                            "std_dev_seconds": std_dev,
                            "tokens_per_second": avg_tokens_per_second,
                            "samples": num_samples
                        })
                        
                        print(f"  Average: {avg_time:.3f} seconds (±{std_dev:.3f})")
                        print(f"  Performance: {avg_tokens_per_second:.1f} tokens/second")
                        
                        success_count += 1
                        
                    except Exception as e:
                        print(f"  Error at sequence length {seq_length}, batch size {batch_size}: {e}")
                        error_count += 1
                        # Continue with next configuration rather than failing completely
                        continue
            
            print(f"\nLLM Benchmark Summary: {success_count} configurations succeeded, {error_count} failed")
            
            return success_count > 0
        
        except Exception as e:
            print(f"Error benchmarking LLM: {e}")
            print("\nTrying to provide more diagnostic information:")
            import traceback
            traceback.print_exc()
            return False

    def save_results(self, filename=None):
        """Save benchmark results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gpu_name = self.gpu_name.replace(" ", "_").lower()
            filename = f"{self.output_dir}/benchmark_{gpu_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        return filename

def main():
    parser = argparse.ArgumentParser(description='GPU Benchmarking Tool')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU name (for labeling results)')
    parser.add_argument('--sd-model', type=str, default="stabilityai/stable-diffusion-2-1",
                        help='Stable Diffusion model ID')
    parser.add_argument('--llm-model', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='LLM model ID')
    parser.add_argument('--resolutions', type=str, default="512x512,768x768,1024x1024",
                        help='Comma-separated list of resolutions (WxH)')
    parser.add_argument('--steps', type=str, default="20,50",
                        help='Comma-separated list of step counts')
    parser.add_argument('--seq-lengths', type=str, default="128,512,1024",
                        help='Comma-separated list of sequence lengths')
    parser.add_argument('--batch-sizes', type=str, default="1,4",
                        help='Comma-separated list of batch sizes')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of samples to average')
    parser.add_argument('--output-dir', type=str, default="benchmark_results",
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Parse lists
    resolutions = [tuple(map(int, r.split('x'))) for r in args.resolutions.split(',')]
    steps = list(map(int, args.steps.split(',')))
    seq_lengths = list(map(int, args.seq_lengths.split(',')))
    batch_sizes = list(map(int, args.batch_sizes.split(',')))
    
    # Initialize benchmark
    benchmark = GPUBenchmark(gpu_name=args.gpu, output_dir=args.output_dir)
    
    # Run benchmarks
    sd_success = benchmark.benchmark_stable_diffusion(
        model_id=args.sd_model,
        resolutions=resolutions,
        step_counts=steps,
        num_samples=args.samples
    )
    
    llm_success = benchmark.benchmark_llm(
        model_id=args.llm_model,
        sequence_lengths=seq_lengths,
        batch_sizes=batch_sizes,
        num_samples=args.samples
    )
    
    # Save results
    benchmark.save_results()

if __name__ == "__main__":
    main()
