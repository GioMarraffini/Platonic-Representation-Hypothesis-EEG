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
    We extract embeddings from ALL 12 transformer encoder layers using forward hooks.
    This enables N×M layer comparisons across models for the Platonic
    Representation Hypothesis analysis.

    Output shape: (n_layers, n_channels, n_patches, 200) = (12, n_ch, n_patches, 200)
    First dimension is always the number of layers.

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
from typing import Optional, List
from .base import BaseExtractor
from .registry import register_extractor


@register_extractor("cbramod")
class CBraModExtractor(BaseExtractor):
    """
    Extractor for CBraMod using official pretrained weights.
    
    The pretrained model expects:
    - Input shape: (batch, ch_num, patch_num, patch_size=200)
    - patch_size=200 samples (1 second at 200 Hz)
    
    Embeddings are extracted from ALL 12 transformer encoder layers using forward hooks.
    Output shape: (n_layers, n_ch, n_patches, 200) = (12, n_ch, n_patches, 200)
    First dimension is layers, enabling layer-wise comparisons.
    """
    
    MODEL_NAME = "cbramod"
    REQUIRED_SFREQ = 200.0  # CBraMod expects 200 Hz
    EMBEDDING_DIM = 200  # Output embedding dimension
    PATCH_SIZE = 200  # 1 second at 200 Hz
    NUM_LAYERS = 12  # Number of transformer encoder layers
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(device=device, layer_name=layer_name, verbose=verbose)
        self._layer_outputs: List[torch.Tensor] = []
        self._hooks = []
    
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
            
            # Register hooks on all encoder layers to capture outputs
            self._register_layer_hooks()
            
            self._is_loaded = True
            
            if self.verbose:
                print(f"CBraMod loaded successfully!")
                print(f"  - Embedding dimension: {self.EMBEDDING_DIM}")
                print(f"  - Number of layers: {self.NUM_LAYERS}")
                print(f"  - Patch size: {self.PATCH_SIZE} samples")
                print(f"  - Expected sfreq: {self.REQUIRED_SFREQ} Hz")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CBraMod pretrained model: {e}")
    
    def _register_layer_hooks(self) -> None:
        """Register forward hooks on all encoder layers to capture outputs."""
        def make_hook(layer_idx: int):
            def hook(module, input, output):
                # output shape: (batch, ch_num, patch_num, d_model)
                self._layer_outputs.append(output.detach())
            return hook
        
        # CBraMod encoder has self.model.encoder.layers which is a ModuleList
        for i, layer in enumerate(self.model.encoder.layers):
            hook = layer.register_forward_hook(make_hook(i))
            self._hooks.append(hook)
        
        if self.verbose:
            print(f"  Registered hooks on {len(self._hooks)} encoder layers")
    
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
        
        Captures outputs from ALL 12 transformer encoder layers using forward hooks,
        enabling N×M layer comparisons across models.
        
        Returns:
            Embedding of shape (n_layers, n_channels, n_patches, 200) = (12, n_ch, n_patches, 200)
            First dimension is layers, enabling layer-wise comparisons.
        """
        # Prepare input
        x = self._prepare_input(raw)  # (1, n_channels, n_patches, 200)
        
        if self.verbose:
            print(f"  Input shape: {x.shape}")
        
        # Clear previous layer outputs
        self._layer_outputs = []
        
        # Forward pass (hooks capture all layer outputs)
        with torch.no_grad():
            _ = self.model(x)
        
        # Stack all layer outputs
        # Each layer output is (1, ch, patches, 200)
        layer_embeddings = []
        for layer_out in self._layer_outputs:
            # Remove batch dimension: (ch, patches, 200)
            layer_emb = layer_out.squeeze(0).cpu().numpy()
            layer_embeddings.append(layer_emb)
        
        # Stack into (n_layers, n_ch, n_patches, 200)
        embedding = np.stack(layer_embeddings, axis=0)
        
        if self.verbose:
            print(f"  Output embedding shape: {embedding.shape}")
        
        return embedding
