import os
import torch
import gc
import logging
import sys
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_model_available(model_name: str, model_class, cache_dir: Optional[str] = None) -> str:
    """Ensure model is available locally, downloading if necessary."""
    try:
        try:
            model_class.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            logger.info(f"Model {model_name} found in cache")
            return model_name
        except Exception:
            logger.info(f"Downloading {model_name}...")
            model_class.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
            return model_name
    except Exception as e:
        logger.error(f"Model availability check failed: {str(e)}")
        sys.exit(1)

def validate_path(file_path: str) -> bool:
    """Validate if path exists and is a file."""
    normalized = os.path.normpath(file_path.strip().replace('"', '').replace("'", ''))
    return os.path.isfile(normalized)

def load_custom_clip(g_path: str, l_path: str) -> CLIPTextModel:
    """Load and merge CLIP models with collision resolution and validation."""
    try:
        state_dict_g = load_file(g_path)
        state_dict_l = load_file(l_path)
        
        merged_state = {}
        conflict_count = 0
        all_keys = set(state_dict_g.keys()).union(state_dict_l.keys())
        
        for key in all_keys:
            tensor_g = state_dict_g.get(key)
            tensor_l = state_dict_l.get(key)
            
            if tensor_g is not None and tensor_l is not None:
                if tensor_g.shape != tensor_l.shape:
                    logger.warning(f"Skipping key {key} due to shape mismatch: {tensor_g.shape} vs {tensor_l.shape}")
                    conflict_count += 1
                    continue
                merged_state[key] = (tensor_g + tensor_l) / 2
            else:
                merged_state[key] = tensor_g if tensor_g is not None else tensor_l
        
        logger.info(f"Merged CLIP models with {conflict_count} key conflicts resolved")
        
        config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPTextModel(config)
        load_result = model.load_state_dict(merged_state, strict=False)
        
        if load_result.missing_keys:
            logger.warning(f"Missing keys in CLIP merge: {', '.join(load_result.missing_keys[:3])}...")
        if load_result.unexpected_keys:
            logger.warning(f"Unexpected keys in CLIP merge: {', '.join(load_result.unexpected_keys[:3])}...")
        
        return model
    except Exception as e:
        logger.error(f"CLIP loading failed: {str(e)}")
        sys.exit(1)

def fft_blend(
    param1: torch.Tensor,
    param2: torch.Tensor,
    blend_coefficients: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 512
) -> torch.Tensor:
    """Optimized FFT-based parameter blending with memory-efficient chunking."""
    orig_shape = param1.shape
    device = param1.device
    dtype = param1.dtype
    
    # Flatten to 2D tensor for batch processing
    param1_flat = param1.view(-1, orig_shape[-1])
    param2_flat = param2.view(-1, orig_shape[-1])
    mask_flat = mask.view(-1, orig_shape[-1])
    
    blended = torch.zeros_like(param1_flat, device=device)
    
    # Pre-calculate frequency coefficients
    freq_coeff = blend_coefficients[:, :param1_flat.size(-1)//2+1].float()
    
    for i in range(0, param1_flat.size(0), chunk_size):
        chunk1 = param1_flat[i:i+chunk_size].float()
        chunk2 = param2_flat[i:i+chunk_size].float()
        current_chunk_size = chunk1.size(0)
        
        # Generate frequency-domain blend coefficients
        coeff = freq_coeff[:current_chunk_size].to(device)
        
        # FFT processing
        fft1 = torch.fft.rfft(chunk1, dim=-1)
        fft2 = torch.fft.rfft(chunk2, dim=-1)
        
        # Complex number blending
        blended_real = coeff * fft1.real + (1 - coeff) * fft2.real
        blended_imag = coeff * fft1.imag + (1 - coeff) * fft2.imag
        blended_fft = torch.complex(blended_real, blended_imag)
        
        # Inverse FFT and mask application
        blended_chunk = torch.fft.irfft(blended_fft, n=chunk1.size(-1))
        mask_chunk = mask_flat[i:i+chunk_size].to(device)
        blended_chunk = torch.where(mask_chunk, (chunk1 + chunk2) / 2, blended_chunk)
        
        # Store result with original dtype
        blended[i:i+chunk_size] = blended_chunk.to(dtype)
    
    return blended.view(orig_shape)

class QuantumMerger:
    """Main merger class with state management and optimization."""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.hypernet = self._init_hypernet().to(self.device)
        logger.info(f"Initialized QuantumMerger on {device} with {self.hypernet}")
        
    def _init_hypernet(self) -> torch.nn.Module:
        """Initialize optimized hypernetwork architecture."""
        return torch.nn.Sequential(
            torch.nn.Linear(768, 1536),
            torch.nn.GELU(),
            torch.nn.Linear(1536, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, 384),
            torch.nn.Tanh()
        )
    
    def generate_blend_coefficients(self, text_emb: torch.Tensor) -> torch.Tensor:
        """Generate frequency blending coefficients from text embeddings."""
        with torch.no_grad():
            return self.hypernet(text_emb.to(self.device))
    
    @staticmethod
    def create_mask(shape: Tuple[int], seed: int, threshold: float) -> torch.Tensor:
        """Create reproducible decoherence mask with exact threshold matching."""
        generator = torch.Generator(device='cpu').manual_seed(seed)
        rand_values = torch.rand(shape, generator=generator)
        return (rand_values < threshold).to(torch.bool)
    
    def _prepare_text_embeddings(
        self,
        prompt: str,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel
    ) -> torch.Tensor:
        """Process text prompt into embeddings."""
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            return outputs.last_hidden_state.mean(dim=1)
    
    def merge(
        self,
        model1: Dict[str, torch.Tensor],
        model2: Dict[str, torch.Tensor],
        prompt: str,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        entanglement_strength: float = 0.7714,
        decoherence_factor: float = 0.2,
        chunk_size: int = 4096
    ) -> Dict[str, torch.Tensor]:
        """Core merging algorithm with memory optimization."""
        text_emb = self._prepare_text_embeddings(prompt, tokenizer, text_encoder)
        blend_coeff = self.generate_blend_coefficients(text_emb)
        
        merged_model = {}
        keys = sorted(set(model1.keys()).intersection(model2.keys()))
        logger.info(f"Merging {len(keys)} common parameters")
        
        progress_bar = tqdm(keys, desc="Merging parameters")
        for key in progress_bar:
            param1 = model1[key].to(self.device, non_blocking=True)
            param2 = model2[key].to(self.device, non_blocking=True)
            
            if param1.shape != param2.shape:
                logger.warning(f"Shape mismatch {param1.shape} vs {param2.shape} for {key}")
                merged_model[key] = ((param1 + param2) / 2).cpu()
                continue
                
            if 'weight' in key and len(param1.shape) > 1:
                seed = abs(hash(prompt + key)) % (2**32)
                mask = self.create_mask(param1.shape, seed, decoherence_factor)
                
                try:
                    blended = fft_blend(
                        param1, param2,
                        blend_coeff,
                        mask,
                        chunk_size
                    )
                    merged = blended * entanglement_strength + \
                            (param1 + param2) * (1 - entanglement_strength) / 2
                except RuntimeError as e:
                    logger.error(f"FFT blend failed for {key}: {str(e)}")
                    merged = (param1 + param2) / 2
            else:
                merged = (param1 + param2) / 2
                
            merged_model[key] = merged.half().cpu()
            
            # Memory cleanup
            del param1, param2, merged
            if 'weight' in key:
                del blended
            torch.cuda.empty_cache()
            
        return merged_model

def quantum_merge_models(
    model1_path: str,
    model2_path: str,
    prompt: str,
    output_path: str,
    entanglement_strength: float = 0.7714,
    decoherence_factor: float = 0.2,
    chunk_size: int = 4096
):
    """Main entry point for model merging."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing merge process on {device.upper()}")
        
        # Load models with memory mapping
        logger.info("Loading model files...")
        model1 = load_file(model1_path)
        model2 = load_file(model2_path)
        
        # Initialize merger
        merger = QuantumMerger(device=device)
        
        # Load CLIP components
        clip_model_name = "openai/clip-vit-large-patch14"
        ensure_model_available(clip_model_name, CLIPTextModel)
        tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device).eval()
        
        # Perform merge
        logger.info("Starting merge process...")
        merged_model = merger.merge(
            model1, model2, prompt,
            tokenizer, text_encoder,
            entanglement_strength,
            decoherence_factor,
            chunk_size
        )
        
        # Save result
        save_file(merged_model, output_path)
        logger.info(f"Successfully saved merged model to {output_path}")
        
    except Exception as e:
        logger.error(f"Critical merge error: {str(e)}")
        sys.exit(1)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Interactive command-line interface."""
    print("\n=== AI Model Quantum Merger ===")
    print("===============================\n")
    
    # Input validation
    def get_valid_path(prompt: str) -> str:
        while True:
            path = input(prompt).strip()
            if validate_path(path):
                return os.path.normpath(path)
            print(f"Invalid path: {path}. Please try again.")
    
    # Get input paths
    base_model = get_valid_path("Enter path to base model: ")
    secondary_model = get_valid_path("Enter path to secondary model: ")
    
    # Output path handling
    while True:
        output_path = os.path.normpath(input("Enter output path: ").strip())
        output_dir = os.path.dirname(output_path)
        if not output_dir:
            print("Invalid output path")
            continue
        try:
            os.makedirs(output_dir, exist_ok=True)
            break
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
    
    # Get prompt
    prompt = input("Enter merge guidance prompt: ").strip()
    while not prompt:
        print("Prompt cannot be empty")
        prompt = input("Enter merge guidance prompt: ").strip()
    
    # Advanced parameters
    print("\nAdvanced Parameters (press Enter for defaults)")
    try:
        entanglement = float(input("Entanglement strength (0.0-1.0) [0.7714]: ") or 0.7714)
        decoherence = float(input("Decoherence factor (0.0-1.0) [0.2]: ") or 0.2)
        chunk = int(input("Processing chunk size [4096]: ") or 4096)
    except ValueError:
        print("Invalid input, using defaults")
        entanglement = 0.7714
        decoherence = 0.2
        chunk = 4096
    
    # Start merging
    print("\nStarting merge process...")
    quantum_merge_models(
        model1_path=base_model,
        model2_path=secondary_model,
        prompt=prompt,
        output_path=output_path,
        entanglement_strength=entanglement,
        decoherence_factor=decoherence,
        chunk_size=chunk
    )

if __name__ == "__main__":
    main()
