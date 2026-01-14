"""Generate synthetic EEG data in .fif format for testing."""

import mne
import numpy as np
from pathlib import Path
from typing import Optional, List


# Standard 10-20 channel names
STANDARD_10_20 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]


def generate_synthetic_eeg(
    n_channels: int = 19,
    duration: float = 10.0,
    sfreq: float = 256.0,
    channel_names: Optional[List[str]] = None,
    add_noise: bool = True,
    add_alpha: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic EEG-like data with realistic frequency components.
    
    Args:
        n_channels: Number of EEG channels
        duration: Duration in seconds
        sfreq: Sampling frequency in Hz
        channel_names: List of channel names (uses standard 10-20 if None)
        add_noise: Whether to add pink noise
        add_alpha: Whether to add alpha band oscillation (8-12 Hz)
        seed: Random seed for reproducibility
        
    Returns:
        Synthetic EEG data of shape (n_channels, n_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq
    
    # Initialize with low-amplitude white noise
    data = np.random.randn(n_channels, n_samples) * 5  # µV scale
    
    if add_noise:
        # Add 1/f (pink) noise characteristic of EEG
        freqs = np.fft.rfftfreq(n_samples, 1/sfreq)
        freqs[0] = 1  # Avoid division by zero
        pink_spectrum = 1 / np.sqrt(freqs)
        
        for ch in range(n_channels):
            white = np.random.randn(n_samples)
            fft_white = np.fft.rfft(white)
            pink_fft = fft_white * pink_spectrum
            pink = np.fft.irfft(pink_fft, n=n_samples)
            data[ch] += pink * 10  # Scale to µV
    
    if add_alpha:
        # Add alpha oscillation (8-12 Hz) especially in posterior channels
        alpha_freq = 10 + np.random.randn() * 0.5  # ~10 Hz
        alpha_amplitude = 20  # µV
        alpha_wave = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
        
        # Stronger in posterior channels (O1, O2, P3, P4, Pz)
        posterior_weight = np.array([
            0.3 if i < 7 else  # Frontal
            0.5 if i < 12 else  # Central/Temporal
            0.8 if i < 17 else  # Parietal
            1.0  # Occipital
            for i in range(n_channels)
        ])
        
        for ch in range(n_channels):
            phase = np.random.rand() * 2 * np.pi
            data[ch] += posterior_weight[ch] * alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t + phase)
    
    # Add some random bursts (mimicking transient events)
    n_bursts = np.random.randint(2, 5)
    for _ in range(n_bursts):
        burst_start = np.random.randint(0, n_samples - int(sfreq * 0.5))
        burst_len = int(sfreq * (0.1 + np.random.rand() * 0.3))
        burst_ch = np.random.randint(0, n_channels)
        burst_freq = 15 + np.random.rand() * 10  # Beta range
        t_burst = np.arange(burst_len) / sfreq
        burst = 15 * np.sin(2 * np.pi * burst_freq * t_burst) * np.hanning(burst_len)
        data[burst_ch, burst_start:burst_start + burst_len] += burst
    
    return data


def generate_synthetic_fif_data(
    output_dir: str = "data",
    n_files: int = 3,
    n_channels: int = 19,
    duration: float = 10.0,
    sfreq: float = 256.0,
    prefix: str = "synthetic_eeg",
    seed: Optional[int] = 42,
) -> List[Path]:
    """
    Generate multiple synthetic .fif files for testing.
    
    Args:
        output_dir: Directory to save .fif files
        n_files: Number of files to generate
        n_channels: Number of EEG channels
        duration: Duration in seconds
        sfreq: Sampling frequency in Hz
        prefix: Prefix for file names
        seed: Base random seed (incremented for each file)
        
    Returns:
        List of paths to generated .fif files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use standard 10-20 channel names
    channel_names = STANDARD_10_20[:n_channels]
    if n_channels > len(STANDARD_10_20):
        # Add extra channels if needed
        for i in range(n_channels - len(STANDARD_10_20)):
            channel_names.append(f"EEG{i+1}")
    
    generated_files = []
    
    for i in range(n_files):
        file_seed = seed + i if seed is not None else None
        
        # Generate synthetic data
        data = generate_synthetic_eeg(
            n_channels=n_channels,
            duration=duration,
            sfreq=sfreq,
            channel_names=channel_names,
            seed=file_seed,
        )
        
        # Create MNE Info object
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=sfreq,
            ch_types=['eeg'] * n_channels,
        )
        
        # Set standard montage for channel positions
        montage = mne.channels.make_standard_montage('standard_1020')
        
        # Create Raw object
        raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # Convert µV to V
        
        # Set montage (only for channels that exist in montage)
        try:
            raw.set_montage(montage, on_missing='ignore')
        except Exception:
            pass  # Montage setting is optional
        
        # Save to .fif file
        filename = f"{prefix}_{i:03d}_raw.fif"
        filepath = output_dir / filename
        raw.save(filepath, overwrite=True, verbose=False)
        
        generated_files.append(filepath)
        print(f"Generated: {filepath}")
    
    return generated_files


if __name__ == "__main__":
    # Generate test data when run directly
    files = generate_synthetic_fif_data(
        output_dir="data",
        n_files=5,
        n_channels=19,
        duration=10.0,
        sfreq=256.0,
    )
    print(f"\nGenerated {len(files)} synthetic .fif files")

