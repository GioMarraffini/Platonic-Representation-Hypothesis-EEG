# PRH+EEG: EEG Embedding Extraction Pipeline

Extract embeddings from EEG data using various foundation models for representation analysis (RSA, MKNN, Platonic Representation Hypothesis).

## Overview

This repository provides a unified pipeline to extract embeddings from `.fif` EEG files using:

### EEG Foundation Models
- **LaBraM** - Large Brain Model (via braindecode)
- **CBraMod** - Criss-Cross Brain Modality Transformer

### Time Series Foundation Models
- **Chronos** - Amazon's time series forecasting model (tiny/mini/small/base/large)
- **Moirai** - Salesforce's universal time series model (small/base/large)
- **Moirai-MoE** - Moirai with Mixture of Experts
- **Time-MoE** - Time series Mixture of Experts model (base/large)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

```

## Project Structure

```
PRH+EEG/
├── data/                    # Input .fif files
├── representations/         # Output embeddings (organized by model)
│   ├── labram/
│   ├── chronos/
│   ├── cbramod/
│   └── ...
├── models/                  # Model-specific extractors
│   ├── base.py              # Base extractor class
│   ├── registry.py          # Model registry
│   ├── labram_extractor.py
│   ├── chronos_extractor.py
│   └── cbramod_extractor.py
├── utils/                   # Utility functions
│   ├── fif_utils.py         # .fif file loading
│   └── synthetic.py         # Synthetic data generation
├── extract_embeddings.py    # Main CLI pipeline
└── requirements.txt
```

## Usage

### Basic Usage

```bash
# Extract embeddings using LaBraM
python extract_embeddings.py --model labram --input data/ --output representations/

# Use Chronos (time series model)
python extract_embeddings.py --model chronos --input data/ --model-size small

# Use CBraMod
python extract_embeddings.py --model cbramod --input data/

# Use GPU
python extract_embeddings.py --model labram --input data/ --device cuda
```

### Process a Single File

```bash
python extract_embeddings.py --model labram --input data/subject_001.fif
```

### List Available Models

```bash
python extract_embeddings.py --list-models
```

### Generate Synthetic Test Data

```python
from utils.synthetic import generate_synthetic_fif_data

# Generate 5 synthetic .fif files
files = generate_synthetic_fif_data(
    output_dir="data",
    n_files=5,
    n_channels=19,
    duration=10.0,
    sfreq=256.0,
)
```

## Output Format

For each `.fif` file, the pipeline saves a `.npy` file with the same base name:

- Input: `data/subject_001_raw.fif`
- Output: `representations/labram/subject_001.npy`

**Embedding shapes:**
- **LaBraM**: `(200,)` - single vector per file
- **CBraMod**: `(n_channels, n_patches, 200)` - preserves spatial-temporal structure
- **Chronos**: `(n_channels, context_length, d_model)` - preserves channel and temporal structure

For RSA analysis, embeddings must have consistent shape **within** each model across files.

## Python API

```python
from models import get_extractor
from utils import load_fif

# Initialize extractor
extractor = get_extractor("labram", device="cuda", verbose=True)

# Load and process
raw = load_fif("data/subject_001.fif")
embeddings = extractor.extract_embeddings(raw)

print(f"Embedding shape: {embeddings.shape}")
```

### Process Multiple Files

```python
from pathlib import Path
from models import get_extractor
from utils import get_fif_files

extractor = get_extractor("chronos", model_size="small")

for fif_path in get_fif_files("data/"):
    embeddings = extractor.process_file(
        fif_path,
        output_dir=Path("representations/chronos")
    )
```

## Representational Analysis

Each model outputs embeddings with consistent shape across files (within that model). For RSA, you compare representations within each model separately:

### Representational Similarity Analysis (RSA)

```python
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path

# Load embeddings (flatten multi-dimensional embeddings for comparison)
labram_dir = Path("representations/labram")
labram_embs = [np.load(f).flatten() for f in sorted(labram_dir.glob("*.npy"))]

# Compute RDM
def compute_rdm(embeddings):
    n = len(embeddings)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = cosine(embeddings[i], embeddings[j])
    return rdm

rdm = compute_rdm(labram_embs)
```

## Model Details

### LaBraM
- **Input**: EEG at 200 Hz (channels padded/truncated to 128)
- **Output**: `(200,)` - single vector per file
- **Embedding source**: fc_norm layer (before classifier)
- **Reference**: [LaBraM Paper](https://github.com/935963004/LaBraM)

### CBraMod
- **Input**: Multi-channel EEG at 200 Hz
- **Output**: `(n_channels, n_patches, 200)` - preserves spatial-temporal structure
- **Embedding source**: Encoder output (before proj_out)
- **Reference**: [CBraMod](https://github.com/wjq-learning/CBraMod)

### Chronos
- **Input**: Univariate time series (processes each EEG channel independently)
- **Output**: `(n_channels, context_length, d_model)` - preserves channel and temporal structure
- **d_model by size**: tiny=256, mini=384, small=512, base=768, large=1024
- **Embedding source**: Encoder hidden states via `embed()` method
- **Reference**: [Chronos](https://github.com/amazon-science/chronos-forecasting)

## License

MIT License

