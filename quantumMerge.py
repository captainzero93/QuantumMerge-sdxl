import torch
import gc
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel

clip_g              = "E:\Apps\...\clip_clip_g_00001_.safetensors"
clip_l              = "E:\Apps\...\clip_clip_l_00001_.safetensors"
prompt_to_follow    = "1girl, spiderman"
base_model          = "E:\Apps\...\base_model.safetensors"
secondary_model     = "E:\Apps\...\secondary_model.safetensors"
output_path         = "E:\Apps\...\output.safetensors"

def load_custom_clip(g_path, l_path):
    # Load both safetensors files
    state_dict_g = load_file(g_path)
    state_dict_l = load_file(l_path)
    
    # Merge state dictionaries
    merged_state = {**state_dict_g, **state_dict_l}
    
    # Load original CLIP configuration
    config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
    
    # Create CLIP model instance
    text_encoder = CLIPTextModel(config)
    
    # Load custom weights
    load_result = text_encoder.load_state_dict(merged_state, strict=False)
    
    # Print warnings for debugging
    if len(load_result.missing_keys) > 0:
        print(f"Missing keys: {load_result.missing_keys}")
    if len(load_result.unexpected_keys) > 0:
        print(f"Unexpected keys: {load_result.unexpected_keys}")
    
    return text_encoder

def process_fft_chunked(param1_half, param2_half, hyper_out, decoherence_mask, chunk_size=32):
    orig_shape = param1_half.shape
    flat_shape = (-1, orig_shape[-1])
    flat1 = param1_half.view(flat_shape)
    flat2 = param2_half.view(flat_shape)
    flat_mask = decoherence_mask.view(flat_shape)
    processed_chunks = []
    
    for i in tqdm(range(0, flat1.shape[0], chunk_size), desc="Processing FFT chunks", leave=False):
        with torch.no_grad():  # Disable gradient tracking
            # Move chunks to GPU and convert to float32
            chunk1 = flat1[i:i+chunk_size].float()
            chunk2 = flat2[i:i+chunk_size].float()
            mask_chunk = flat_mask[i:i+chunk_size].to('cuda', non_blocking=True)

            # FFT processing
            fft1 = torch.fft.rfft(chunk1, dim=-1)
            fft2 = torch.fft.rfft(chunk2, dim=-1)
            freq_dim = fft1.shape[-1]

            # Prepare coefficients
            if hyper_out.shape[-1] < freq_dim:
                coeff = hyper_out.repeat(1, freq_dim // hyper_out.shape[-1] + 1)[:, :freq_dim]
            else:
                coeff = hyper_out[:, :freq_dim]
            coeff = coeff.expand(chunk1.size(0), -1).float()

            # Frequency blending
            magnitude_blend = torch.sigmoid(coeff * 5)
            phase_blend = torch.sigmoid(coeff * 3 - 1)

            blended_fft_real = magnitude_blend * fft1.real + (1 - magnitude_blend) * fft2.real
            blended_fft_imag = phase_blend * fft1.imag + (1 - phase_blend) * fft2.imag
            blended_fft = torch.complex(blended_fft_real, blended_fft_imag)

            # Inverse FFT and mask application
            blended_chunk = torch.fft.irfft(blended_fft, n=chunk1.shape[-1], dim=-1)
            avg = (chunk1 + chunk2) / 2
            blended_chunk[mask_chunk] = avg[mask_chunk]

            # Move data back to CPU and clean up
            blended_chunk = blended_chunk.half().cpu()
            processed_chunks.append(blended_chunk)

            # Explicit cleanup
            del chunk1, chunk2, fft1, fft2, blended_fft, avg, mask_chunk, magnitude_blend, phase_blend, coeff
    
    blended_flat = torch.cat(processed_chunks, dim=0)
    return blended_flat.view(orig_shape)

def quantum_merge_models(model1_path, model2_path, prompt, output_path,
                         entanglement_strength=0.7714, decoherence_factor=0.2,
                         chunk_size=4096):
    # Load models with memory mapping
    model1 = load_file(model1_path)
    model2 = load_file(model2_path)

    # Hypernetwork with reduced precision
    hypernet = torch.nn.Sequential(
        torch.nn.Linear(768, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 256),
        torch.nn.Tanh()
    ).cuda().half()

    with torch.no_grad():
        # Load tokenizer and custom CLIP
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = load_custom_clip(
            clip_g,
            clip_l
        ).to("cuda").eval()

        # Tokenize and encode prompt (rest of your existing code here)
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to("cuda")
        
        # Generate text embeddings
        text_outputs = text_encoder(text_input_ids)
        text_emb = text_outputs.pooler_output
        text_emb = text_emb.half()

        # Generate hypernetwork parameters
        hyper_out = hypernet(text_emb).float()

        merged_model = {}
        keys = list(model1.keys())
        
        for key in tqdm(keys, desc="Merging parameters"):
            if key in model2:
                param1 = model1[key].cuda().half()
                param2 = model2[key].cuda().half()

                if 'weight' in key:
                    # Generate mask on CPU
                    seed = abs(hash(prompt + key)) % (2**32)
                    torch.manual_seed(seed)
                    decoherence_mask = torch.rand(param1.shape, device='cpu') < decoherence_factor

                    # Process with memory optimization
                    blended = process_fft_chunked(param1, param2, hyper_out, decoherence_mask, chunk_size)

                    # Entanglement blending on CPU
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
                gc.collect()  # Force garbage collection

            else:
                merged_model[key] = model1[key]

    save_file(merged_model, output_path)
    del model1, model2, hypernet, hyper_out, text_emb
    gc.collect()

quantum_merge_models(
    model1_path=base_model,
    model2_path=secondary_model,
    prompt=prompt_to_follow,
    output_path=output_path
)
