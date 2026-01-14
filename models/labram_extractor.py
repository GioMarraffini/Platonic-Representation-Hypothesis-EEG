"""
LaBraM (Large Brain Model) extractor using braindecode.

Architecture:
    Input: (batch, channels, time_samples)
    → Patch Embedding (Conv1d segmentation + temporal convs)
    → Positional Embedding + CLS token
    → 12 Transformer Blocks (self-attention + MLP)
    → Mean Pooling + LayerNorm (fc_norm) → 200-dim EMBEDDING
    → Classification Head

Embedding extraction:
    We extract embeddings from ALL 12 transformer blocks using forward hooks.
    This enables N×M layer comparisons across models for the Platonic
    Representation Hypothesis analysis.

    Output shape: (n_layers, embedding_dim) = (12, 200)
    First dimension is always the number of layers.

Reference:
    - Paper: "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI"
    - GitHub: https://github.com/935963004/LaBraM
    - Pretrained: braindecode/labram-pretrained
"""

import numpy as np
import mne
import torch
from typing import Optional, List
from .base import BaseExtractor
from .registry import register_extractor


@register_extractor("labram")
class LaBraMExtractor(BaseExtractor):
    """
    Extractor for LaBraM using braindecode's pretrained model.
    
    The pretrained model expects:
    - 128 channels (but can work with fewer by padding)
    - 200 Hz sampling rate
    - Input shape: (batch, channels, time_samples)
    
    Embeddings are extracted from ALL 12 transformer blocks using forward hooks.
    Output shape: (n_layers, 200) = (12, 200) where first dimension is layers.
    """
    
    MODEL_NAME = "labram"
    REQUIRED_SFREQ = 200.0  # LaBraM expects 200 Hz
    EMBEDDING_DIM = 200  # Output embedding dimension
    NUM_LAYERS = 12  # Number of transformer blocks
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(device=device, layer_name=layer_name, verbose=verbose)
        self.n_chans_pretrained = 128  # Pretrained model has 128 channels
        self.n_times_pretrained = 3000  # 15 seconds at 200 Hz
        self._layer_outputs: List[torch.Tensor] = []
        self._hooks = []
    
    def load_model(self) -> None:
        """Load LaBraM pretrained model from HuggingFace via braindecode."""
        try:
            from braindecode.models import Labram
            
            if self.verbose:
                print("Loading LaBraM pretrained model from braindecode/labram-pretrained...")
            
            # Load pretrained model
            self.model = Labram.from_pretrained("braindecode/labram-pretrained")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Register hooks on all transformer blocks
            self._register_layer_hooks()
            
            self._is_loaded = True
            
            if self.verbose:
                print(f"LaBraM loaded successfully!")
                print(f"  - Embedding dimension: {self.EMBEDDING_DIM}")
                print(f"  - Number of layers: {self.NUM_LAYERS}")
                print(f"  - Expected channels: {self.n_chans_pretrained}")
                print(f"  - Expected sfreq: {self.REQUIRED_SFREQ} Hz")
            
        except ImportError as e:
            raise ImportError(
                f"braindecode is required for LaBraM. Install with: pip install braindecode\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LaBraM pretrained model: {e}")
    
    def _register_layer_hooks(self) -> None:
        """Register forward hooks on all transformer blocks to capture layer outputs."""
        def make_hook(layer_idx: int):
            def hook(module, input, output):
                # output shape: (batch, seq_len, embed_dim)
                self._layer_outputs.append(output.detach())
            return hook
        
        # LaBraM has self.model.blocks which is a ModuleList of transformer blocks
        for i, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(make_hook(i))
            self._hooks.append(hook)
        
        if self.verbose:
            print(f"  Registered hooks on {len(self._hooks)} transformer blocks")
    
    def _prepare_input(self, raw: mne.io.Raw) -> torch.Tensor:
        """
        Prepare input tensor from raw EEG data.
        
        - Picks EEG channels
        - Pads/truncates channels to match pretrained model
        - Segments into windows if needed
        
        Returns:
            Tensor of shape (n_windows, n_channels, n_times)
        """
        # Get EEG data
        raw_eeg = raw.copy().pick(picks="eeg", exclude="bads")
        data = raw_eeg.get_data()  # (n_channels, n_samples)
        n_channels, n_samples = data.shape
        
        # Normalize data (z-score)
        data = (data - data.mean()) / (data.std() + 1e-8)
        
        # Pad channels if needed (pretrained expects 128 channels)
        if n_channels < self.n_chans_pretrained:
            padded = np.zeros((self.n_chans_pretrained, n_samples))
            padded[:n_channels, :] = data
            data = padded
        elif n_channels > self.n_chans_pretrained:
            # Truncate to 128 channels
            data = data[:self.n_chans_pretrained, :]
        
        # Segment into windows of n_times_pretrained samples (with 50% overlap)
        window_size = self.n_times_pretrained
        step = window_size // 2
        
        windows = []
        start = 0
        while start + window_size <= n_samples:
            window = data[:, start:start + window_size]
            windows.append(window)
            start += step
        
        # If no complete windows, pad the data
        if len(windows) == 0:
            padded = np.zeros((self.n_chans_pretrained, window_size))
            padded[:, :min(n_samples, window_size)] = data[:, :min(n_samples, window_size)]
            windows.append(padded)
        
        # Stack into tensor
        x = np.stack(windows, axis=0)  # (n_windows, n_channels, n_times)
        return torch.tensor(x, dtype=torch.float32).to(self.device)
    
    def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extract embeddings from raw EEG data using LaBraM.
        
        Captures outputs from ALL 12 transformer blocks using forward hooks,
        enabling N×M layer comparisons across models.
        
        Returns:
            Embedding of shape (n_layers, 200) = (12, 200)
            First dimension is layers, enabling layer-wise comparisons.
        """
        # Prepare input
        x = self._prepare_input(raw)  # (n_windows, 128, n_times)
        
        if self.verbose:
            print(f"  Input shape: {x.shape}")
        
        # Collect layer outputs across all windows
        all_layer_embeddings = [[] for _ in range(self.NUM_LAYERS)]
        
        with torch.no_grad():
            for i in range(0, len(x), 8):  # Batch of 8
                batch = x[i:i+8]
                
                # Clear previous layer outputs
                self._layer_outputs = []
                
                # Forward pass triggers all hooks
                _ = self.model.forward_features(batch)
                
                # Process layer outputs
                # Each output is (batch, seq_len, embed_dim)
                # We apply mean pooling to get (batch, embed_dim) per layer
                for layer_idx, layer_out in enumerate(self._layer_outputs):
                    # Mean pool across sequence dimension (same as forward_features does at the end)
                    pooled = layer_out.mean(dim=1)  # (batch, embed_dim)
                    all_layer_embeddings[layer_idx].append(pooled.cpu().numpy())
        
        # Concatenate windows and average for each layer
        layer_embeddings = []
        for layer_idx in range(self.NUM_LAYERS):
            # Concatenate all batches: (total_windows, embed_dim)
            layer_windows = np.concatenate(all_layer_embeddings[layer_idx], axis=0)
            # Average across windows: (embed_dim,)
            layer_avg = layer_windows.mean(axis=0)
            layer_embeddings.append(layer_avg)
        
        # Stack into (n_layers, embed_dim)
        embedding = np.stack(layer_embeddings, axis=0)  # (12, 200)
        
        if self.verbose:
            print(f"  Output embedding shape: {embedding.shape}")
        
        return embedding
