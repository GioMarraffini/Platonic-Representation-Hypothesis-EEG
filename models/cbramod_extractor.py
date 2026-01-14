"""
CBraMod (Criss-Cross Brain Modality) extractor using official code.

Architecture:
    Input: (batch, ch_num, patch_num, patch_size) → e.g., (batch, 19, 10, 200)
    → Patch Embedding (Conv2d + FFT spectral features + positional encoding)
    → 12 Criss-Cross Transformer Layers:
        - First half of embedding: Spatial attention (across channels)
        - Second half: Temporal attention (across patches)
    → Output Projection (Linear) → (batch, ch, patches, 200)

Embedding extraction:
    We extract from the encoder output and mean pool across channels and patches
    to get a 200-dimensional representation per file.

Reference:
    - Paper: "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding"
    - GitHub: https://github.com/wjq-learning/CBraMod
    - Pretrained: weighting666/CBraMod/pretrained_weights.pth
"""

import numpy as np
import mne
import torch
import sys
from pathlib import Path
from typing import Optional
from .base import BaseExtractor
from .registry import register_extractor


@register_extractor("cbramod")
class CBraModExtractor(BaseExtractor):
    """
    Extractor for CBraMod using official pretrained weights.
    
    The pretrained model expects:
    - Input shape: (batch, ch_num, patch_num, patch_size=200)
    - patch_size=200 samples (1 second at 200 Hz)
    
    Embedding is extracted from encoder output and mean pooled:
    - encoder output: (batch, ch_num, patch_num, 200)
    - mean pool → (batch, 200) → (200,) per file
    """
    
    MODEL_NAME = "cbramod"
    REQUIRED_SFREQ = 200.0  # CBraMod expects 200 Hz
    EMBEDDING_DIM = 200  # Output embedding dimension
    PATCH_SIZE = 200  # 1 second at 200 Hz
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(device=device, layer_name=layer_name, verbose=verbose)
        self._encoder_output = None
    
    def load_model(self) -> None:
        """Load CBraMod pretrained model from HuggingFace using official code."""
        try:
            from huggingface_hub import hf_hub_download
            
            if self.verbose:
                print("Loading CBraMod pretrained weights from weighting666/CBraMod...")
            
            # Load official CBraMod model code
            cbramod_repo = Path(__file__).parent.parent / "CBraMod_repo"
            if cbramod_repo.exists():
                # Use importlib to avoid conflicts with our models package
                import importlib.util
                
                # Load criss_cross_transformer first (dependency)
                cc_spec = importlib.util.spec_from_file_location(
                    "criss_cross_transformer",
                    cbramod_repo / "models" / "criss_cross_transformer.py"
                )
                cc_module = importlib.util.module_from_spec(cc_spec)
                sys.modules["criss_cross_transformer"] = cc_module
                cc_spec.loader.exec_module(cc_module)
                
                # Patch the import in cbramod.py
                import types
                fake_models = types.ModuleType("models")
                fake_models.criss_cross_transformer = cc_module
                sys.modules["models"] = fake_models
                sys.modules["models.criss_cross_transformer"] = cc_module
                
                # Load cbramod
                cbramod_spec = importlib.util.spec_from_file_location(
                    "cbramod_official",
                    cbramod_repo / "models" / "cbramod.py"
                )
                cbramod_module = importlib.util.module_from_spec(cbramod_spec)
                cbramod_spec.loader.exec_module(cbramod_module)
                
                CBraMod = cbramod_module.CBraMod
                
                if self.verbose:
                    print("Using official CBraMod model code from cloned repo")
            else:
                raise FileNotFoundError(
                    f"CBraMod repo not found at {cbramod_repo}. "
                    "Please clone it: git clone https://github.com/wjq-learning/CBraMod.git CBraMod_repo"
                )
            
            # Download pretrained weights
            weights_path = hf_hub_download(
                repo_id="weighting666/CBraMod",
                filename="pretrained_weights.pth",
            )
            
            # Initialize model with default config (matching pretrained)
            self.model = CBraMod(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=30, n_layer=12, nhead=8
            )
            
            # Load pretrained weights
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Register hook on encoder to capture output before proj_out
            self._register_encoder_hook()
            
            self._is_loaded = True
            
            if self.verbose:
                print(f"CBraMod loaded successfully!")
                print(f"  - Embedding dimension: {self.EMBEDDING_DIM}")
                print(f"  - Patch size: {self.PATCH_SIZE} samples")
                print(f"  - Expected sfreq: {self.REQUIRED_SFREQ} Hz")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CBraMod pretrained model: {e}")
    
    def _register_encoder_hook(self):
        """Register hook to capture encoder output before proj_out."""
        def hook(module, input, output):
            self._encoder_output = output.detach()
        
        # Hook on the encoder (the TransformerEncoder)
        self.model.encoder.register_forward_hook(hook)
    
    def _prepare_input(self, raw: mne.io.Raw) -> torch.Tensor:
        """
        Prepare input tensor from raw EEG data.
        
        CBraMod expects: (batch, ch_num, patch_num, patch_size)
        where patch_size=200 (1 second at 200 Hz)
        
        Returns:
            Tensor of shape (1, n_channels, n_patches, 200)
        """
        # Get EEG data
        raw_eeg = raw.copy().pick(picks="eeg", exclude="bads")
        data = raw_eeg.get_data()  # (n_channels, n_samples)
        n_channels, n_samples = data.shape
        
        # Normalize data (z-score per channel)
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        
        # Segment into patches of PATCH_SIZE samples
        n_patches = n_samples // self.PATCH_SIZE
        
        if n_patches == 0:
            # Pad if too short
            padded = np.zeros((n_channels, self.PATCH_SIZE))
            padded[:, :min(n_samples, self.PATCH_SIZE)] = data[:, :min(n_samples, self.PATCH_SIZE)]
            patches = padded.reshape(n_channels, 1, self.PATCH_SIZE)
        else:
            # Truncate to complete patches
            data = data[:, :n_patches * self.PATCH_SIZE]
            patches = data.reshape(n_channels, n_patches, self.PATCH_SIZE)
        
        # Shape: (1, n_channels, n_patches, patch_size)
        x = patches[np.newaxis, ...]  # Add batch dimension
        
        return torch.tensor(x, dtype=torch.float32).to(self.device)
    
    def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extract embeddings from raw EEG data using CBraMod.
        
        Uses encoder output (before proj_out), preserving full spatial-temporal structure.
        
        Returns:
            Embedding of shape (n_channels, n_patches, 200) - preserves structure
            For RSA, this full tensor is compared across files.
        """
        # Prepare input
        x = self._prepare_input(raw)  # (1, n_channels, n_patches, 200)
        
        if self.verbose:
            print(f"  Input shape: {x.shape}")
        
        # Forward pass (hook captures encoder output)
        with torch.no_grad():
            _ = self.model(x)
        
        # Get encoder output (before proj_out)
        if self._encoder_output is not None:
            feats = self._encoder_output  # (1, ch, patches, 200)
        else:
            # Fallback: use model output
            feats = self.model(x)
        
        # Remove batch dimension, keep full (n_channels, n_patches, 200) structure
        embedding = feats.squeeze(0).cpu().numpy()  # (n_channels, n_patches, 200)
        
        if self.verbose:
            print(f"  Output embedding shape: {embedding.shape}")
        
        return embedding
