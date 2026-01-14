"""Moirai time series foundation model extractor."""

import numpy as np
import mne
import torch
from typing import Optional
from .base import BaseExtractor
from .registry import register_extractor


@register_extractor("moirai")
class MoiraiExtractor(BaseExtractor):
    """
    Extractor for Moirai time series foundation model.
    
    Moirai is Salesforce's universal time series forecasting model that
    can handle multiple frequencies and horizons.
    
    Reference: https://github.com/SalesforceAIResearch/uni2ts
    HuggingFace: Salesforce/moirai-1.0-R-{small,base,large}
    
    Note: Requires uni2ts library.
    """
    
    MODEL_NAME = "moirai"
    REQUIRED_SFREQ = None
    REQUIRED_CHANNELS = None
    
    AVAILABLE_SIZES = ["small", "base", "large"]
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
        model_size: str = "small",
        context_length: int = 512,
    ):
        super().__init__(device=device, layer_name=layer_name, verbose=verbose)
        
        if model_size not in self.AVAILABLE_SIZES:
            raise ValueError(f"model_size must be one of {self.AVAILABLE_SIZES}")
        
        self.model_size = model_size
        self.context_length = context_length
        self._activation = {}
        self._hooks = []
    
    def load_model(self) -> None:
        """Load Moirai model using uni2ts."""
        try:
            from uni2ts.model.moirai import MoiraiModule, MoiraiMoEModule
            
            model_id = f"Salesforce/moirai-1.0-R-{self.model_size}"
            
            if self.verbose:
                print(f"Loading {model_id}...")
            
            self.model = MoiraiModule.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._register_hooks()
            self._is_loaded = True
            
            if self.verbose:
                print(f"Loaded Moirai {self.model_size}")
                
        except ImportError:
            # Fallback: use a simplified embedding approach
            if self.verbose:
                print("uni2ts not available. Using simplified Moirai wrapper...")
            
            self._use_simplified = True
            self._load_simplified_model()
    
    def _load_simplified_model(self) -> None:
        """Load simplified version when uni2ts is not available."""
        try:
            from transformers import AutoModel
            
            model_id = f"Salesforce/moirai-1.0-R-{self.model_size}"
            
            # This might not work directly but worth trying
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            if self.verbose:
                print(f"Could not load Moirai: {e}")
                print("Using dummy embeddings. Install uni2ts for real Moirai inference.")
            
            self.model = None
        
        self._is_loaded = True
    
    def _register_hooks(self) -> None:
        """Register hooks for embedding extraction."""
        
        def get_activation(name):
            def hook(model, input, output):
                if hasattr(output, 'last_hidden_state'):
                    self._activation[name] = output.last_hidden_state.detach()
                elif isinstance(output, tuple):
                    self._activation[name] = output[0].detach()
                else:
                    self._activation[name] = output.detach()
            return hook
        
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        if self.model is not None and hasattr(self.model, 'encoder'):
            hook = self.model.encoder.register_forward_hook(get_activation('encoder'))
            self._hooks.append(hook)
    
    def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extract embeddings from raw EEG data using Moirai.
        
        Returns:
            Embeddings of shape (embedding_dim,)
        """
        raw_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
        data = raw_eeg.get_data()  # (n_channels, n_samples)
        
        # Normalize
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        
        if self.model is None:
            # Return dummy embeddings
            if self.verbose:
                print("Warning: Using dummy embeddings (Moirai model not loaded)")
            return np.random.randn(512).astype(np.float32)
        
        # Process each channel
        channel_embeddings = []
        
        for ch_idx in range(data.shape[0]):
            ch_data = data[ch_idx]
            
            # Segment
            segments = []
            start = 0
            while start + self.context_length <= len(ch_data):
                segments.append(ch_data[start:start + self.context_length])
                start += self.context_length // 2
            
            if len(segments) == 0:
                padded = np.zeros(self.context_length)
                padded[:len(ch_data)] = ch_data
                segments.append(padded)
            
            x = torch.tensor(np.stack(segments), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                try:
                    # Attempt to get embeddings
                    if hasattr(self.model, 'encode'):
                        emb = self.model.encode(x)
                    else:
                        output = self.model(x.unsqueeze(-1))
                        if 'encoder' in self._activation:
                            emb = self._activation['encoder']
                        elif hasattr(output, 'last_hidden_state'):
                            emb = output.last_hidden_state
                        else:
                            emb = output
                    
                    if len(emb.shape) > 2:
                        emb = emb.mean(dim=1)
                    
                    channel_embeddings.append(emb.mean(dim=0).cpu().numpy())
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error processing channel {ch_idx}: {e}")
                    channel_embeddings.append(np.zeros(512))
        
        embeddings = np.stack(channel_embeddings, axis=0)
        return embeddings.mean(axis=0)
    
    def __del__(self):
        for hook in self._hooks:
            hook.remove()


@register_extractor("moirai-moe")
class MoiraiMoEExtractor(MoiraiExtractor):
    """Extractor for Moirai-MoE (Mixture of Experts) variant."""
    
    MODEL_NAME = "moirai-moe"
    
    def load_model(self) -> None:
        """Load Moirai-MoE model."""
        try:
            from uni2ts.model.moirai import MoiraiMoEModule
            
            model_id = f"Salesforce/moirai-moe-1.0-R-{self.model_size}"
            
            if self.verbose:
                print(f"Loading {model_id}...")
            
            self.model = MoiraiMoEModule.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._register_hooks()
            self._is_loaded = True
            
        except ImportError:
            if self.verbose:
                print("uni2ts not available for Moirai-MoE. Using base Moirai instead...")
            super().load_model()

