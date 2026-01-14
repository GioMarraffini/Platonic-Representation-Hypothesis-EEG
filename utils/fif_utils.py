"""Utilities for loading and handling .fif EEG files."""

import mne
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional


def load_fif(fif_path: Union[str, Path], verbose: bool = False) -> mne.io.Raw:
    """
    Load a .fif file using MNE.
    
    Args:
        fif_path: Path to the .fif file
        verbose: Whether to print MNE loading messages
        
    Returns:
        MNE Raw object
    """
    mne.set_log_level("WARNING" if not verbose else "INFO")
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=verbose)
    return raw


def get_fif_files(data_dir: Union[str, Path]) -> List[Path]:
    """
    Get all .fif files from a directory.
    
    Args:
        data_dir: Directory containing .fif files
        
    Returns:
        List of paths to .fif files
    """
    data_dir = Path(data_dir)
    fif_files = list(data_dir.glob("*.fif"))
    return sorted(fif_files)


def raw_to_epochs(
    raw: mne.io.Raw,
    epoch_duration: float = 2.0,
    overlap: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert raw EEG data to fixed-length epochs.
    
    Args:
        raw: MNE Raw object
        epoch_duration: Duration of each epoch in seconds
        overlap: Overlap between epochs in seconds
        
    Returns:
        Tuple of (epochs_data, channel_names)
        epochs_data shape: (n_epochs, n_channels, n_samples)
    """
    # Pick only EEG channels
    raw_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
    
    sfreq = raw_eeg.info['sfreq']
    data = raw_eeg.get_data()
    n_channels, n_total_samples = data.shape
    
    samples_per_epoch = int(epoch_duration * sfreq)
    step = int((epoch_duration - overlap) * sfreq)
    
    epochs = []
    start = 0
    while start + samples_per_epoch <= n_total_samples:
        epoch = data[:, start:start + samples_per_epoch]
        epochs.append(epoch)
        start += step
    
    if len(epochs) == 0:
        # If data is too short, pad it
        padded = np.zeros((n_channels, samples_per_epoch))
        padded[:, :min(n_total_samples, samples_per_epoch)] = data[:, :min(n_total_samples, samples_per_epoch)]
        epochs.append(padded)
    
    epochs_data = np.stack(epochs, axis=0)
    channel_names = raw_eeg.ch_names
    
    return epochs_data, channel_names


def resample_raw(raw: mne.io.Raw, target_sfreq: float) -> mne.io.Raw:
    """
    Resample raw data to target sampling frequency.
    
    Args:
        raw: MNE Raw object
        target_sfreq: Target sampling frequency
        
    Returns:
        Resampled MNE Raw object
    """
    if raw.info['sfreq'] != target_sfreq:
        raw = raw.copy().resample(target_sfreq)
    return raw

