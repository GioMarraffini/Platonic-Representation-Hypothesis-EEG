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

# Optional: Install uni2ts for Moirai models
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
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
│   ├── cbramod_extractor.py
│   ├── moirai_extractor.py
│   └── timemoe_extractor.py
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

The embedding shape depends on the model but is typically `(embedding_dim,)` - a single vector per file, suitable for representational similarity analysis.

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

Since all models output embeddings with the same dimensionality per file, you can easily compute:

### Representational Similarity Analysis (RSA)

```python
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path

# Load embeddings from two models
labram_dir = Path("representations/labram")
chronos_dir = Path("representations/chronos")

labram_embs = [np.load(f) for f in sorted(labram_dir.glob("*.npy"))]
chronos_embs = [np.load(f) for f in sorted(chronos_dir.glob("*.npy"))]

# Compute RDMs (Representational Dissimilarity Matrices)
def compute_rdm(embeddings):
    n = len(embeddings)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = cosine(embeddings[i], embeddings[j])
    return rdm

rdm_labram = compute_rdm(labram_embs)
rdm_chronos = compute_rdm(chronos_embs)

# Compare RDMs (Spearman correlation of upper triangles)
from scipy.stats import spearmanr
triu_idx = np.triu_indices(len(labram_embs), k=1)
r, p = spearmanr(rdm_labram[triu_idx], rdm_chronos[triu_idx])
print(f"RSA correlation: r={r:.3f}, p={p:.4f}")
```

## Model Details

### LaBraM
- **Input**: 19-channel EEG at 200 Hz
- **Output**: 256-dimensional embedding
- **Reference**: [LaBraM Paper](https://github.com/935963004/LaBraM)

### Chronos
- **Input**: Univariate time series (processes each EEG channel independently)
- **Output**: 512-dimensional embedding (T5-based)
- **Sizes**: tiny, mini, small, base, large
- **Reference**: [Chronos](https://github.com/amazon-science/chronos-forecasting)

### CBraMod
- **Input**: Multi-channel EEG at 200 Hz
- **Output**: 256-dimensional embedding
- **Reference**: [CBraMod](https://github.com/wjq-learning/CBraMod)

### Moirai
- **Input**: Univariate time series
- **Output**: Variable-dimensional embedding
- **Sizes**: small, base, large
- **Reference**: [Moirai](https://github.com/SalesforceAIResearch/uni2ts)

## License

MIT License

