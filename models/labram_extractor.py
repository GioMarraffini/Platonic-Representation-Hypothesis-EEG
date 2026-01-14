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
    We use the `forward_features()` method which returns the 200-dimensional
    representation AFTER mean pooling and fc_norm, BEFORE the classification head.
    This is the standard approach for ViT-like models.

Reference:
    - Paper: "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI"
    - GitHub: https://github.com/935963004/LaBraM
    - Pretrained: braindecode/labram-pretrained
"""

import numpy as np
import mne
import torch
from typing import Optional
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
    
    Embedding is extracted from fc_norm layer (200-dim) using forward_features().
    """
    
    MODEL_NAME = "labram"
    REQUIRED_SFREQ = 200.0  # LaBraM expects 200 Hz
    EMBEDDING_DIM = 200  # Output embedding dimension
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(device=device, layer_name=layer_name, verbose=verbose)
        self.n_chans_pretrained = 128  # Pretrained model has 128 channels
        self.n_times_pretrained = 3000  # 15 seconds at 200 Hz
    
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
            
            self._is_loaded = True
            
            if self.verbose:
                print(f"LaBraM loaded successfully!")
                print(f"  - Embedding dimension: {self.EMBEDDING_DIM}")
                print(f"  - Expected channels: {self.n_chans_pretrained}")
                print(f"  - Expected sfreq: {self.REQUIRED_SFREQ} Hz")
            
        except ImportError as e:
            raise ImportError(
                f"braindecode is required for LaBraM. Install with: pip install braindecode\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LaBraM pretrained model: {e}")
    
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
        
        Uses the forward_features() method to get the 200-dim representation
        from fc_norm (before the classification head).
        
        Returns:
            Embedding of shape (200,) - single vector per file
        """
        # Prepare input
        x = self._prepare_input(raw)  # (n_windows, 128, n_times)
        
        if self.verbose:
            print(f"  Input shape: {x.shape}")
        
        # Extract embeddings using forward_features
        embeddings_list = []
        
        with torch.no_grad():
            for i in range(0, len(x), 8):  # Batch of 8
                batch = x[i:i+8]
                
                # forward_features returns the embedding before classification head
                emb = self.model.forward_features(batch)  # (batch, 200)
                embeddings_list.append(emb.cpu().numpy())
        
        # Concatenate all window embeddings
        embeddings = np.concatenate(embeddings_list, axis=0)  # (n_windows, 200)
        
        # Average across windows to get single representation per file
        embedding = embeddings.mean(axis=0)  # (200,)
        
        if self.verbose:
            print(f"  Output embedding shape: {embedding.shape}")
        
        return embedding
