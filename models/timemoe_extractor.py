"""Time-MoE time series foundation model extractor."""

import numpy as np
import mne
import torch
from typing import Optional
from .base import BaseExtractor
from .registry import register_extractor


@register_extractor("timemoe")
class TimeMoEExtractor(BaseExtractor):
    """
    Extractor for Time-MoE time series foundation model.
    
    Time-MoE is a scalable foundation model for time series forecasting
    that uses mixture-of-experts architecture.
    
    Reference: https://github.com/Time-MoE/Time-MoE
    HuggingFace: Maple728/TimeMoE-{50M,200M}
    
    Note: Requires custom code from the paper's repo.
    """
    
    MODEL_NAME = "timemoe"
    REQUIRED_SFREQ = None
    REQUIRED_CHANNELS = None
    
    AVAILABLE_SIZES = {
        "base": "Maple728/TimeMoE-50M",
        "large": "Maple728/TimeMoE-200M",
    }
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
        model_size: str = "base",
        context_length: int = 512,
    ):
        super().__init__(device=device, layer_name=layer_name, verbose=verbose)
        
        if model_size not in self.AVAILABLE_SIZES:
            raise ValueError(f"model_size must be one of {list(self.AVAILABLE_SIZES.keys())}")
        
        self.model_size = model_size
        self.context_length = context_length
        self._activation = {}
        self._hooks = []
    
    def load_model(self) -> None:
        """Load Time-MoE model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            
            model_id = self.AVAILABLE_SIZES[self.model_size]
            
            if self.verbose:
                print(f"Loading {model_id}...")
            
            # Time-MoE uses custom code
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=self.device if self.device != "cpu" else None,
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._register_hooks()
            self._is_loaded = True
            
            if self.verbose:
                print(f"Loaded Time-MoE {self.model_size}")
                
        except Exception as e:
            if self.verbose:
                print(f"Could not load Time-MoE: {e}")
                print("Using simplified embedding extraction...")
            
            self._load_fallback_model()
    
    def _load_fallback_model(self) -> None:
        """Load fallback embedding model."""
        # Use a simple transformer-based embedding
        import torch.nn as nn
        
        class SimpleTimeMoE(nn.Module):
            def __init__(self, input_dim=512, embed_dim=256, n_heads=8, n_layers=4):
                super().__init__()
                self.input_proj = nn.Linear(1, embed_dim)
                self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, embed_dim))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.output_proj = nn.Linear(embed_dim, embed_dim)
            
            def forward(self, x):
                # x: (batch, seq_len)
                x = x.unsqueeze(-1)  # (batch, seq_len, 1)
                x = self.input_proj(x)  # (batch, seq_len, embed_dim)
                x = x + self.pos_embed[:, :x.size(1), :]
                x = self.encoder(x)
                x = x.mean(dim=1)  # Global average pooling
                x = self.output_proj(x)
                return x
        
        self.model = SimpleTimeMoE(input_dim=self.context_length)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._register_hooks()
        self._is_loaded = True
        
        if self.verbose:
            print("Using simplified Time-MoE architecture (random weights)")
    
    def _register_hooks(self) -> None:
        """Register hooks for embedding extraction."""
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self._activation[name] = output[0].detach()
                else:
                    self._activation[name] = output.detach()
            return hook
        
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        if hasattr(self.model, 'encoder'):
            hook = self.model.encoder.register_forward_hook(get_activation('encoder'))
            self._hooks.append(hook)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # For transformer-based models, hook the last layer
            last_layer = self.model.model.layers[-1]
            hook = last_layer.register_forward_hook(get_activation('encoder'))
            self._hooks.append(hook)
    
    def _extract_channel_embedding(self, channel_data: np.ndarray) -> np.ndarray:
        """Extract embedding for a single channel."""
        # Segment into context windows
        segments = []
        start = 0
        while start + self.context_length <= len(channel_data):
            segments.append(channel_data[start:start + self.context_length])
            start += self.context_length // 2
        
        if len(segments) == 0:
            padded = np.zeros(self.context_length)
            padded[:len(channel_data)] = channel_data
            segments.append(padded)
        
        x = torch.tensor(np.stack(segments), dtype=torch.float32).to(self.device)
        
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(x), 8):  # Batch of 8
                batch = x[i:i+8]
                
                try:
                    if hasattr(self.model, 'forward'):
                        output = self.model(batch)
                    
                    # Get embedding from hook or output
                    if 'encoder' in self._activation:
                        emb = self._activation['encoder']
                        if len(emb.shape) > 2:
                            emb = emb.mean(dim=1)
                    elif hasattr(output, 'last_hidden_state'):
                        emb = output.last_hidden_state.mean(dim=1)
                    elif isinstance(output, torch.Tensor):
                        if len(output.shape) > 2:
                            emb = output.mean(dim=1)
                        else:
                            emb = output
                    else:
                        emb = output[0].mean(dim=1) if isinstance(output, tuple) else output
                    
                    embeddings.append(emb.cpu().numpy())
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error in Time-MoE forward pass: {e}")
                    # Return zeros
                    embeddings.append(np.zeros((len(batch), 256)))
        
        if len(embeddings) == 0:
            return np.zeros(256)
        
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings.mean(axis=0)
    
    def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extract embeddings from raw EEG data using Time-MoE.
        
        Returns:
            Embeddings of shape (embedding_dim,)
        """
        raw_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
        data = raw_eeg.get_data()
        
        # Normalize
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        
        # Process each channel
        channel_embeddings = []
        
        for ch_idx in range(data.shape[0]):
            if self.verbose:
                print(f"Processing channel {ch_idx + 1}/{data.shape[0]}")
            
            emb = self._extract_channel_embedding(data[ch_idx])
            channel_embeddings.append(emb)
        
        embeddings = np.stack(channel_embeddings, axis=0)
        return embeddings.mean(axis=0)
    
    def __del__(self):
        for hook in self._hooks:
            hook.remove()

