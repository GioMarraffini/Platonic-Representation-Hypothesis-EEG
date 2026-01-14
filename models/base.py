"""Base class for all embedding extractors."""

import numpy as np
import mne
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union


class BaseExtractor(ABC):
    """
    Abstract base class for EEG embedding extractors.
    
    Each model-specific extractor should inherit from this class and implement:
    - load_model(): Load the pretrained model
    - extract_embeddings(): Extract embeddings from raw EEG data
    
    The base class handles:
    - Model caching
    - Output saving
    - Common preprocessing steps
    """
    
    # Class attributes to be overridden by subclasses
    MODEL_NAME: str = "base"
    REQUIRED_SFREQ: Optional[float] = None  # Required sampling frequency, None = any
    REQUIRED_CHANNELS: Optional[int] = None  # Required number of channels, None = any
    
    def __init__(
        self,
        device: str = "cpu",
        layer_name: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the extractor.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            layer_name: Specific layer to extract embeddings from (model-specific)
            verbose: Whether to print verbose output
        """
        self.device = device
        self.layer_name = layer_name
        self.verbose = verbose
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the pretrained model.
        
        This method should set self.model and self._is_loaded = True
        """
        pass
    
    @abstractmethod
    def extract_embeddings(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extract embeddings from raw EEG data.
        
        Args:
            raw: MNE Raw object containing EEG data
            
        Returns:
            numpy array of embeddings. The FIRST DIMENSION is always the number
            of layers (for multi-layer extraction to enable N×M comparisons).
            
            Typical shapes:
            - (n_layers, embedding_dim) for LaBraM: (12, 200)
            - (n_layers, n_channels, n_patches, embedding_dim) for CBraMod: (12, n_ch, n_patches, 200)
            - (n_layers, n_channels, seq_len, d_model) for time series models
        """
        pass
    
    def preprocess(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Preprocess raw data for the model.
        
        Override this method in subclasses for model-specific preprocessing.
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Preprocessed MNE Raw object
        """
        raw = raw.copy()
        
        # Resample if required
        if self.REQUIRED_SFREQ is not None:
            if raw.info['sfreq'] != self.REQUIRED_SFREQ:
                raw = raw.resample(self.REQUIRED_SFREQ)
        
        return raw
    
    def ensure_loaded(self) -> None:
        """Ensure the model is loaded before extraction."""
        if not self._is_loaded:
            if self.verbose:
                print(f"Loading {self.MODEL_NAME} model...")
            self.load_model()
            self._is_loaded = True
    
    def process_file(
        self,
        fif_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Process a single .fif file and optionally save embeddings.
        
        Args:
            fif_path: Path to the .fif file
            output_dir: Directory to save embeddings (optional)
            
        Returns:
            numpy array of embeddings
        """
        self.ensure_loaded()
        
        fif_path = Path(fif_path)
        
        if self.verbose:
            print(f"Processing: {fif_path.name}")
        
        # Load and preprocess
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        raw = self.preprocess(raw)
        
        # Extract embeddings
        embeddings = self.extract_embeddings(raw)
        
        # Save if output directory specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Replace .fif extension with .npy
            output_name = fif_path.stem
            if output_name.endswith("_raw"):
                output_name = output_name[:-4]  # Remove _raw suffix
            output_path = output_dir / f"{output_name}.npy"
            
            np.save(output_path, embeddings)
            
            if self.verbose:
                print(f"Saved: {output_path} (shape: {embeddings.shape})")
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.MODEL_NAME,
            "required_sfreq": self.REQUIRED_SFREQ,
            "required_channels": self.REQUIRED_CHANNELS,
            "device": self.device,
            "layer_name": self.layer_name,
            "is_loaded": self._is_loaded,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.MODEL_NAME}, device={self.device})"

