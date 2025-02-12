import os
import torch
import gc
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import sys
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import psutil
import platform
from pathlib import Path

def get_available_memory():
    """
    Get available GPU and system memory.
    
    Returns:
        tuple: (gpu_memory_available, system_memory_available) in bytes
    """
    # Get GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_allocated = torch.cuda.memory_allocated(0)
        gpu_memory_available = gpu_memory - gpu_memory_allocated
    else:
        gpu_memory_available = 0
    
    # Get system memory
    system_memory = psutil.virtual_memory()
    system_memory_available = system_memory.available
    
    return gpu_memory_available, system_memory_available

def calculate_optimal_chunk_size(param_size):
    """
    Calculate optimal chunk size based on available GPU memory.
    
    Args:
        param_size (int): Size of parameter tensor
    
    Returns:
        int: Optimal chunk size
    """
    gpu_memory, system_memory = get_available_memory()
    
    # Calculate memory needed for processing one chunk
    # We need space for: 2 input chunks, FFT results, blended results, and some overhead
    memory_per_element = 4  # bytes per float32 element
    memory_overhead_factor = 3  # Factor for FFT and intermediate results
    
    if gpu_memory > 0:
        # Use 70% of available GPU memory
        safe_memory = gpu_memory * 0.7
        max_elements = safe_memory / (memory_per_element * memory_overhead_factor)
    else:
        # Use 50% of available system memory if no GPU
        safe_memory = system_memory * 0.5
        max_elements = safe_memory / (memory_per_element * memory_overhead_factor)
    
    # Calculate chunk size
    chunk_size = min(int(max_elements), param_size)
    
    # Round to nearest power of 2 for FFT efficiency
    chunk_size = 2 ** int(torch.log2(torch.tensor(float(chunk_size))).item())
    
    # Ensure chunk size is at least 32 and at most param_size
    return max(32, min(chunk_size, param_size))

def ensure_model_available(model_name, cache_dir=None):
    """
    Ensure the specified model is available, downloading if necessary.
    
    Args:
        model_name (str): Hugging Face model identifier
        cache_dir (str, optional): Custom cache directory
    
    Returns:
        str: Path to the downloaded/cached model
    """
    try:
        if cache_dir:
            model_path = CLIPTextConfig.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False
            )
        else:
            model_path = CLIPTextConfig.from_pretrained(
                model_name, 
                local_files_only=False
            )
        return model_name
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        
        if cache_dir is None:
            home_dir = os.path.expanduser('~')
            cache_dir = os.path.join(home_dir, '.cache', 'huggingface', 'hub')
        
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            
            print(f"Downloading {model_name} to {cache_dir}")
            snapshot_download(
                model_name, 
                cache_dir=cache_dir,
                ignore_patterns=["*.safetensors", "*.bin"]
            )
            
            return model_name
        except Exception as download_error:
            print(f"Failed to download model: {download_error}")
            print("Please check your internet connection and try again.")
            sys.exit(1)

def get_validated_path(prompt, is_output=False):
    """
    Prompt user for a file path and validate it.
    
    Args:
        prompt (str): Prompt message for user input
        is_output (bool): Whether this is an output path
    
    Returns:
        str: Validated file path
    """
    while True:
        file_path = input(prompt).strip().replace('"', '').replace("'", '')
        
        # Convert to Path object for better cross-platform handling
        path = Path(file_path)
        
        if is_output:
            try:
                # Create parent directories if they don't exist
                parent = path.parent
                if not parent.exists():
                    parent.mkdir(parents=True)
                
                # Test write permissions by trying to create/remove a test file
                test_file = parent / '.write_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                    return str(path)
                except (OSError, PermissionError):
                    print(f"Error: No write permission in directory: {parent}")
                    continue
                
            except Exception as e:
                print(f"Error validating output path: {e}")
                continue
        else:
            if path.is_file():
                return str(path)
            else:
                print(f"Error: File not found: {path}")

def load_custom_clip(g_path, l_path):
    """
    Load custom CLIP text encoder from two safetensors files.
    
    Args:
        g_path (str): Path to first safetensors file
        l_path (str): Path to second safetensors file
    
    Returns:
        CLIPTextModel: Loaded and configured text encoder
    """
    try:
        state_dict_g = load_file(g_path)
        state_dict_l = load_file(l_path)
        
        merged_state = {**state_dict_g, **state_dict_l}
        
        config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel(config)
        
        load_result = text_encoder.load_state_dict(merged_state, strict=False)
        
        if len(load_result.missing_keys) > 0:
            print(f"Warning: Missing keys in CLIP model: {load_result.missing_keys}")
        if len(load_result.unexpected_keys) > 0:
            print(f"Warning: Unexpected keys in CLIP model: {load_result.unexpected_keys}")
        
        return text_encoder
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        sys.exit(1)

def process_fft_chunked(param1_half, param2_half, hyper_out, decoherence_mask, chunk_size=None):
    """
    Process model parameters using FFT-based chunked blending.
    
    Args:
        param1_half (torch.Tensor): First model's parameter chunk
        param2_half (torch.Tensor): Second model's parameter chunk
        hyper_out (torch.Tensor): Hypernetwork output
        decoherence_mask (torch.Tensor): Mask for parameter blending
        chunk_size (int, optional): Size of processing chunks
    
    Returns:
        torch.Tensor: Blended parameter chunk
    """
    orig_shape = param1_half.shape
    flat_shape = (-1, orig_shape[-1])
    flat1 = param1_half.view(flat_shape)
    flat2 = param2_half.view(flat_shape)
    flat_mask = decoherence_mask.view(flat_shape)
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(flat1.shape[0])
    
    processed_chunks = []
    
    for i in tqdm(range(0, flat1.shape[0], chunk_size), desc="Processing FFT chunks", leave=False):
        try:
            with torch.no_grad():
                chunk1 = flat1[i:i+chunk_size].float()
                chunk2 = flat2[i:i+chunk_size].float()
                mask_chunk = flat_mask[i:i+chunk_size].to('cuda', non_blocking=True)

                fft1 = torch.fft.rfft(chunk1, dim=-1)
                fft2 = torch.fft.rfft(chunk2, dim=-1)
                freq_dim = fft1.shape[-1]

                if hyper_out.shape[-1] < freq_dim:
                    coeff = hyper_out.repeat(1, freq_dim // hyper_out.shape[-1] + 1)[:, :freq_dim]
                else:
                    coeff = hyper_out[:, :freq_dim]
                coeff = coeff.expand(chunk1.size(0), -1).float()

                magnitude_blend = torch.sigmoid(coeff * 5)
                phase_blend = torch.sigmoid(coeff * 3 - 1)

                blended_fft_real = magnitude_blend * fft1.real + (1 - magnitude_blend) * fft2.real
                blended_fft_imag = phase_blend * fft1.imag + (1 - phase_blend) * fft2.imag
                blended_fft = torch.complex(blended_fft_real, blended_fft_imag)

                blended_chunk = torch.fft.irfft(blended_fft, n=chunk1.shape[-1], dim=-1)
                avg = (chunk1 + chunk2) / 2
                blended_chunk[mask_chunk] = avg[mask_chunk]

                blended_chunk = blended_chunk.half().cpu()
                processed_chunks.append(blended_chunk)

                del chunk1, chunk2, fft1, fft2, blended_fft, avg, mask_chunk
                del magnitude_blend, phase_blend, coeff
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM occurs, clear cache and try with smaller chunk size
                torch.cuda.empty_cache()
                gc.collect()
                chunk_size = chunk_size // 2
                print(f"Reducing chunk size to {chunk_size} due to OOM error")
                if chunk_size < 32:
                    raise RuntimeError("Unable to process with minimum chunk size")
                # Retry this chunk
                i -= chunk_size
                continue
            else:
                raise e
    
    blended_flat = torch.cat(processed_chunks, dim=0)
    return blended_flat.view(orig_shape)

def safe_save_model(model_dict, output_path, max_retries=3):
    """
    Safely save the model with retries and error handling.
    
    Args:
        model_dict (dict): Model state dictionary to save
        output_path (str): Path to save the model
        max_retries (int): Maximum number of save attempts
    """
    for attempt in range(max_retries):
        try:
            save_file(model_dict, output_path)
            print(f"\nMerged model successfully saved to: {output_path}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nError saving model (attempt {attempt + 1}/{max_retries}): {e}")
                print("Retrying...")
                # Clear any potential file handles
                gc.collect()
            else:
                print(f"\nFailed to save model after {max_retries} attempts")
                alt_path = str(Path(output_path).parent / f"emergency_backup_{Path(output_path).name}")
                print(f"Attempting emergency save to: {alt_path}")
                try:
                    save_file(model_dict, alt_path)
                    print(f"Emergency backup saved successfully to: {alt_path}")
                except Exception as backup_error:
                    print(f"Emergency backup also failed: {backup_error}")
                    print("Please check disk space and permissions")
                raise RuntimeError(f"Failed to save model: {e}")

def quantum_merge_models(model1_path, model2_path, prompt, output_path,
                       entanglement_strength=0.7714, decoherence_factor=0.2):
    """
    Merge two AI models using a quantum-inspired blending technique.
    
    Args:
        model1_path (str): Path to first model
        model2_path (str): Path to second model
        prompt (str): Guiding prompt for merging
        output_path (str): Path to save merged model
        entanglement_strength (float, optional): Strength of model blending
        decoherence_factor (float, optional): Randomness factor in blending
    """
    try:
        model1 = load_file(model1_path)
        model2 = load_file(model2_path)

        hypernet = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 256),
            torch.nn.Tanh()
        ).cuda().half()

        with torch.no_grad():
            clip_model_name = "openai/clip-vit-large-patch14"
            ensure_model_available(clip_model_name)
            tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
            
            clip_g = get_validated_path("Enter path to first CLIP safetensors file (clip_g): ")
            clip_l = get_validated_path("Enter path to second CLIP safetensors file (clip_l): ")
            
            text_encoder = load_custom_clip(clip_g, clip_l).to("cuda").eval()

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to("cuda")
            
            text_outputs = text_encoder(text_input_ids)
            text_emb = text_outputs.pooler_output.half()
            hyper_out = hypernet(text_emb).float()

            merged_model = {}
            keys = list(model1.keys())
            
            print("\nStarting model merge process...")
            for key in tqdm(keys, desc="Merging parameters"):
                if key in model2:
                    param1 = model1[key].cuda().half()
                    param2 = model2[key].cuda().half()

                    if 'weight' in key:
                        seed = abs(hash(prompt + key)) % (2**32)
                        torch.manual_seed(seed)
                        decoherence_mask = torch.rand(param1.shape, device='cpu') < decoherence_factor

                        # Use dynamic chunk sizing
                        blended = process_fft_chunked(param1, param2, hyper_out, decoherence_mask)

                        merged = (blended.float() * entanglement_strength +
                                 (param1.cpu().float() * (1 - entanglement_strength) +
                                  param2.cpu().float() * (1 - entanglement_strength)) / 2).half()
                    else:
                        merged = (param1 + param2) / 2

                    merged_model[key] = merged.cpu()
                    
                    # Cleanup GPU memory
                    del param1, param2, merged
                    if 'weight' in key:
                        del blended
                    gc.collect()
                    torch.cuda.empty_cache()

                else:
                    merged_model[key] = model1[key]

            # Save the merged model with error handling
            safe_save_model(merged_model, output_path)

    except Exception as e:
        print(f"\nError during model merging: {e}")
        raise
    finally:
        # Final cleanup
        del model1, model2, hypernet, hyper_out, text_emb
        gc.collect()
        torch.cuda.empty_cache()

def print_system_info():
    """
    Print relevant system information for debugging.
    """
    print("\nSystem Information:")
    print(f"Python version: {platform.python_version()}")
    print(f"Operating System: {platform.system()} {platform.version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {gpu_mem:.2f} GB")
    
    vm = psutil.virtual_memory()
    print(f"System memory: {vm.total / (1024**3):.2f} GB (Available: {vm.available / (1024**3):.2f} GB)")

def validate_parameters(entanglement_strength, decoherence_factor):
    """
    Validate and sanitize input parameters.
    
    Args:
        entanglement_strength (float): Strength of model blending
        decoherence_factor (float): Randomness factor in blending
    
    Returns:
        tuple: Sanitized (entanglement_strength, decoherence_factor)
    """
    # Clamp values to valid ranges
    entanglement_strength = max(0.0, min(1.0, entanglement_strength))
    decoherence_factor = max(0.0, min(1.0, decoherence_factor))
    
    return entanglement_strength, decoherence_factor

def main():
    """
    Main function to interactively merge AI models.
    """
    try:
        print("\nAI Model Quantum Merger")
        print("=" * 50)
        
        print_system_info()
        
        print("\nModel Selection:")
        print("-" * 50)
        
        # Get output filename first
        while True:
            output_filename = input("\nEnter name for the merged model (e.g., 'merged_model.safetensors'): ").strip()
            if not output_filename.endswith('.safetensors'):
                output_filename += '.safetensors'
            
            # Get input model paths
            base_model = get_validated_path("Enter path to base model (first model): ")
            secondary_model = get_validated_path("Enter path to secondary model (second model): ")
            
            # Construct output path in same directory as base model
            output_dir = os.path.dirname(base_model)
            output_path = get_validated_path(
                f"Enter path to save the merged model (default: {os.path.join(output_dir, output_filename)}): ",
                is_output=True
            ) or os.path.join(output_dir, output_filename)
            
            # Only check for file existence if custom path was provided
            if output_path != os.path.join(output_dir, output_filename):
                if os.path.exists(output_path):
                    overwrite = input("Output file already exists. Overwrite? (y/n): ").lower()
                    if overwrite != 'y':
                        continue
            break
        
        # Get guiding prompt
        print("\nMerge Configuration:")
        print("-" * 50)
        while True:
            prompt = input("\nEnter a guiding prompt for model merging\n(e.g., '1girl, detailed anime style, vibrant colors'): ").strip()
            if prompt:
                break
            print("Prompt cannot be empty. Please enter a guiding prompt.")
        
        # Optional: Adjust advanced parameters
        print("\nAdvanced Parameters (press Enter for defaults):")
        print("-" * 50)
        try:
            entanglement_strength = float(input("\nEntanglement Strength (default 0.7714, range 0-1): ") or 0.7714)
            decoherence_factor = float(input("Decoherence Factor (default 0.2, range 0-1): ") or 0.2)
        except ValueError:
            print("\nInvalid input. Using default parameters.")
            entanglement_strength = 0.7714
            decoherence_factor = 0.2
        
        # Validate parameters
        entanglement_strength, decoherence_factor = validate_parameters(
            entanglement_strength, decoherence_factor
        )
        
        print("\nStarting Merge Process:")
        print("-" * 50)
        print(f"\nBase Model: {os.path.basename(base_model)}")
        print(f"Secondary Model: {os.path.basename(secondary_model)}")
        print(f"Output: {output_path}")
        print(f"Entanglement Strength: {entanglement_strength}")
        print(f"Decoherence Factor: {decoherence_factor}")
        print("\nInitiating quantum merge process...")
        
        # Perform model merging
        quantum_merge_models(
            model1_path=base_model,
            model2_path=secondary_model,
            prompt=prompt,
            output_path=output_path,
            entanglement_strength=entanglement_strength,
            decoherence_factor=decoherence_factor
        )
        
        print("\nMerge completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Cleaning up...")
        gc.collect()
        torch.cuda.empty_cache()
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nPlease check the error message above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
