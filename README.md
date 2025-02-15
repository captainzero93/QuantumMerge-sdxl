# Quantum Merger: SDXL Model Blending

Note: This version expects the clip l and g from SD3

## Key Differences from Original Implementation

- **Advanced Architecture**

  - Added a proper transformer-based architecture with self-attention
  - Included positional embeddings for better sequence handling
  - Added multiple transformer blocks for deeper feature extraction
  - Incorporated proper layer normalization and dropout
  - Added residual connections for better gradient flow
  - Introduced `QuantumMerger` class for better code organization

- **Performance Enhancements**
  - More sophisticated FFT-based parameter blending
  - Optimized memory management
  - Improved hypernetwork architecture
  - Efficient chunked processing with better GPU utilization

- **Usability Improvements**
  - Interactive command-line interface
  - Robust input validation
  - Flexible parameter customization
  - Detailed merge process logging
  - Automatic GPU/CPU device selection

- **Debugging Capabilities**
  - Detailed error reporting
  - Progress tracking
  - Explicit handling of model parameter conflicts
  - Implemented comprehensive error handling and logging

## Overview

This script merges two Stable Diffusion models using a novel technique inspired by quantum mechanics principles, including entanglement, decoherence, and frequency-domain processing. It uses a hypernetwork conditioned on a text prompt to guide the merging process, with advanced memory-efficient processing for large models.

## Features

### Quantum-Inspired Merging Techniques
- **Entanglement Blending:** Sophisticated linear interpolation of model parameters
- **Decoherence Mask:** Selective parameter averaging with controllable randomness
- **Frequency Domain Processing:** Advanced FFT-based model parameter blending

### Innovative Capabilities
- **Prompt-Conditioned Hypernetwork:** Text prompt directly influences model merging
- **Advanced Memory Optimization:**
  - Chunked FFT Processing
  - Half-Precision (FP16) Computation
  - Efficient Memory Mapping
  - Aggressive Garbage Collection

### Unique Technical Highlights
- Custom CLIP checkpoint merging and loading
- Reproducible merging process
- Flexible blending coefficient generation

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies
```bash
pip install torch transformers safetensors tqdm
```

### GPU Configuration
Ensure you install the appropriate torch version matching your CUDA configuration:
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/quantum-model-merger.git
cd quantum-model-merger
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script interactively ( AS ADMIN to avoid file save errors ):
```bash
python quantum_model_merger.py
```

The script will guide you through:
- Selecting base and secondary models
- Providing a guidance prompt
- Configuring advanced merging parameters

### Example Prompts

- **Anime Style:** `1girl, detailed anime style, vibrant colors`
- **Photorealistic:** `portrait, cinematic lighting, high detail`
- **Artistic:** `impressionist digital art, soft colors, abstract`

## Advanced Configuration

### Merging Parameters
- `entanglement_strength`: Controls linear interpolation (0.0-1.0)
  - Lower values: More average-based merging
  - Higher values: More FFT-guided blending

- `decoherence_factor`: Manages random parameter averaging (0.0-1.0)
  - Lower values: More deterministic merging
  - Higher values: More random parameter selection
 
Recommended Guidelines:
Style-focused merge:      ent_strength: 0.70-0.75    decoherence: 0.15-0.18
Detail-preserving merge:  ent_strength: 0.65-0.70    decoherence: 0.18-0.22
Experimental merge:       ent_strength: 0.75-0.80    decoherence: 0.22-0.25
Conservative merge:       ent_strength: 0.60-0.65    decoherence: 0.12-0.15

 For initial experiments, start with these defaults and adjust ±0.03 based on your specific use case.
 
- `chunk_size`: Adjusts memory processing chunks
  - Smaller values: Less memory usage, potentially slower
  - Larger values: Faster processing, more memory intensive

## Theoretical Background

### Quantum Mechanics Inspired Concepts

#### Entanglement
- **Metaphorical Interpretation:** Parameters become interconnected
- **Technical Implementation:** Linear interpolation of model weights
- **Controlled by:** `entanglement_strength` parameter

#### Decoherence
- **Metaphorical Interpretation:** Probabilistic parameter blending
- **Technical Implementation:** Random mask-based parameter averaging
- **Controlled by:** `decoherence_factor` parameter

#### Frequency Domain Processing
- **Concept:** Transform parameters to frequency space
- **Benefits:** 
  - Nuanced parameter transformation
  - Prompt-guided blending
  - More intelligent merging strategy

## Limitations and Considerations

- Requires significant GPU memory
- Performance varies with model architectures
- Experimental technique, results may differ
- Primarily tested with Stable Diffusion models

## Troubleshooting

### Common Issues
- **Out of Memory Error:** 
  - Reduce `chunk_size`
  - Close other GPU-intensive applications
  - Use a GPU with more VRAM

- **Model Incompatibility:**
  - Ensure models have similar architectures
  - Check model version compatibility

## Future Roadmap

- [ ] Graphical User Interface (GUI)
- [ ] More sophisticated hypernetwork architectures
- [ ] Layer-specific blending parameters
- [ ] Enhanced visualization of merge results
- [ ] Support for more model types

## Acknowledgments

- Forked from GumGum10
- Inspired by quantum mechanics principles
- Built upon Stable Diffusion and CLIP technologies
- Thanks to the open-source AI community
