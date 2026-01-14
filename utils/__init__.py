"""Utility functions for EEG embedding extraction."""

from .fif_utils import load_fif, get_fif_files
from .synthetic import generate_synthetic_fif_data

__all__ = ["load_fif", "get_fif_files", "generate_synthetic_fif_data"]

