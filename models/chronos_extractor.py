"""
Chronos time series model extractor.

Architecture (T5-based encoder-decoder):
    - Tokenizer: Converts continuous time series to discrete tokens using mean scaling + binning
    - Encoder: T5 encoder processes tokenized sequence
    - Output: Encoder hidden states of shape (batch, seq_len, d_model)

Model variants:
    - chronos-t5-tiny:  d_model=256,  ~8M params
    - chronos-t5-mini:  d_model=384,  ~20M params  
    - chronos-t5-small: d_model=512,  ~46M params
    - chronos-t5-base:  d_model=768,  ~200M params
    - chronos-t5-large: d_model=1024, ~710M params

Embedding extraction:
    Uses the built-in `embed()` method which returns encoder hidden states.
    For EEG: each channel is processed as a univariate time series.
    Output shape: (n_channels, context_length, d_model)

Reference:
    - Paper: "Chronos: Learning the Language of Time Series"
    - GitHub: https://github.com/amazon-science/chronos-forecasting
    - HuggingFace: amazon/chronos-t5-{tiny,mini,small,base,large}
"""

import numpy as np
import mne
import torch
from typing import Optional
from .base import BaseExtractor
from .registry import register_extractor


@register_extractor("chronos")
class ChronosExtractor(BaseExtractor):
    """
    Extractor for Chronos using official ChronosPipeline.embed() method.
    
    Chronos is a univariate time series model, so each EEG channel is
    processed independently as a time series.
    
    Output shape: (n_channels, context_length, d_model)
    - context_length depends on input length (max 512 for standard models)
    - d_model depends on model size (256 for tiny, up to 1024 for large)
    """
    
    MODEL_NAME = "chronos"
    REQUIRED_SFREQ = None  # Flexible, but will resample internally
    
    # Model sizes and their d_model dimensions
    MODEL_SIZES = {
        "tiny": ("amazon/chronos-t5-tiny", 256),
        "mini": ("amazon/chronos-t5-mini", 384),
        "small": ("amazon/chronos-t5-small", 512),
        "base": ("amazon/chronos-t5-base", 768),
        "large": ("amazon/chronos-t5-large", 1024),
    }
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
        model_size: str = "small",
    ):
        """
        Initialize Chronos extractor.
        
        Args:
            model_size: One of 'tiny', 'mini', 'small', 'base', 'large'
        """
        super().__init__(device=device, layer_name=layer_name, verbose=verbose)
        
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"model_size must be one of {list(self.MODEL_SIZES.keys())}")
        
        self.model_size = model_size
        self.model_id, self.d_model = self.MODEL_SIZES[model_size]
        self.context_length = 512  # Default for Chronos
    
    def load_model(self) -> None:
        """Load Chronos model from HuggingFace."""
        try:
            from chronos import ChronosPipeline
            
            if self.verbose:
                print(f"Loading Chronos {self.model_size} from {self.model_id}...")
            
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device if self.device != "cpu" else None,
                dtype=torch.float32,
            )
            
            if self.device == "cpu":
                self.pipeline.model = self.pipeline.model.to("cpu")
            
            self.context_length = self.pipeline.model_context_length
            
            self._is_loaded = True
            
            if self.verbose:
                print(f"Chronos loaded successfully!")
                print(f"  - Model: {self.model_id}")
                print(f"  - d_model: {self.d_model}")
                print(f"  - Context length: {self.context_length}")
            
        except ImportError as e:
            raise ImportError(
                f"chronos-forecasting is required. Install with: pip install chronos-forecasting\n"
                f"Original error: {e}"
            )
    
    def _prepare_input(self, raw: mne.io.Raw) -> torch.Tensor:
        """
        Prepare input tensor from raw EEG data.
        
        Chronos expects univariate time series.
        We process each channel as a separate time series.
        
        Returns:
            Tensor of shape (n_channels, n_samples) where n_samples <= context_length
        """
        # Get EEG data
        raw_eeg = raw.copy().pick(picks="eeg", exclude="bads")
        data = raw_eeg.get_data()  # (n_channels, n_samples)
        n_channels, n_samples = data.shape
        
        # Truncate to context_length if too long
        if n_samples > self.context_length:
            data = data[:, :self.context_length]
        
        # Convert to tensor (Chronos expects float32)
        x = torch.tensor(data, dtype=torch.float32)
        
        return x
    
    def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extract embeddings from raw EEG data using Chronos.
        
        Each channel is processed as an independent time series.
        Uses the official embed() method to get encoder hidden states.
        
        Returns:
            Embedding of shape (n_channels, context_length, d_model)
            Preserves channel and temporal structure.
        """
        # Prepare input
        x = self._prepare_input(raw)  # (n_channels, n_samples)
        n_channels = x.shape[0]
        
        if self.verbose:
            print(f"  Input shape: {x.shape}")
        
        # Process each channel through embed()
        # embed() expects (batch, seq_len) or list of 1D tensors
        with torch.no_grad():
            embeddings, _ = self.pipeline.embed(x)  # (n_channels, seq_len+1, d_model)
        
        # Remove EOS token (last position)
        embeddings = embeddings[:, :-1, :]  # (n_channels, seq_len, d_model)
        
        embedding = embeddings.cpu().numpy()
        
        if self.verbose:
            print(f"  Output embedding shape: {embedding.shape}")
        
        return embedding
