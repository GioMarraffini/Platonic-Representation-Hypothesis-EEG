#!/usr/bin/env python3
"""
Main pipeline script for extracting EEG embeddings using foundation models.

Usage:
    python extract_embeddings.py --model labram --input data/ --output representations/
    python extract_embeddings.py --model chronos --input data/ --output representations/ --device cuda
    python extract_embeddings.py --model cbramod --input data/subject_001.fif --output representations/

Available models:
    - labram: Large Brain Model (EEG foundation model)
    - cbramod: Criss-Cross Brain Modality Transformer
    - chronos: Amazon's time series foundation model
    - moirai: Salesforce's universal time series model
    - moirai-moe: Moirai with Mixture of Experts
    - timemoe: Time-MoE foundation model
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.fif_utils import get_fif_files
from models.registry import ensure_extractors_imported, get_extractor, list_available_models


def extract_all(
    model_name: str,
    input_path: Path,
    output_dir: Path,
    device: str = "cpu",
    layer_name: Optional[str] = None,
    verbose: bool = False,
    **model_kwargs,
) -> List[Path]:
    """
    Extract embeddings for all .fif files using the specified model.
    
    Args:
        model_name: Name of the model to use
        input_path: Path to .fif file or directory containing .fif files
        output_dir: Base output directory (model subfolder will be created)
        device: Device to run inference on ('cpu' or 'cuda')
        layer_name: Specific layer to extract embeddings from
        verbose: Whether to print verbose output
        **model_kwargs: Additional model-specific arguments
        
    Returns:
        List of paths to saved embedding files
    """
    # Ensure extractors are imported
    ensure_extractors_imported()
    
    # Get input files
    input_path = Path(input_path)
    if input_path.is_file():
        fif_files = [input_path]
    else:
        fif_files = get_fif_files(input_path)
    
    if len(fif_files) == 0:
        print(f"No .fif files found in {input_path}")
        return []
    
    print(f"Found {len(fif_files)} .fif file(s)")
    
    # Create output directory for this model
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    print(f"Initializing {model_name} extractor...")
    extractor = get_extractor(
        model_name=model_name,
        device=device,
        layer_name=layer_name,
        verbose=verbose,
        **model_kwargs,
    )
    
    # Load model
    extractor.ensure_loaded()
    print(f"Model loaded: {extractor}")
    
    # Process files
    saved_files = []
    
    for i, fif_path in enumerate(fif_files):
        print(f"\n[{i+1}/{len(fif_files)}] Processing: {fif_path.name}")
        
        try:
            embeddings = extractor.process_file(fif_path, output_dir=model_output_dir)
            
            # Get output path
            output_name = fif_path.stem
            if output_name.endswith("_raw"):
                output_name = output_name[:-4]
            output_path = model_output_dir / f"{output_name}.npy"
            
            saved_files.append(output_path)
            print(f"  -> Saved: {output_path.name} (shape: {embeddings.shape})")
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Completed! Saved {len(saved_files)} embedding files to {model_output_dir}")
    
    return saved_files


def main():
    """Main entry point."""
    # Ensure extractors are imported for listing
    ensure_extractors_imported()
    available_models = list_available_models()
    
    parser = argparse.ArgumentParser(
        description="Extract EEG embeddings using foundation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(available_models)}

Examples:
  python extract_embeddings.py --model labram --input data/
  python extract_embeddings.py --model chronos --input data/ --device cuda
  python extract_embeddings.py --model cbramod --input data/subject_001.fif
  python extract_embeddings.py --model moirai --input data/ --model-size small
""",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        default=None,
        help=f"Model to use for embedding extraction. Options: {', '.join(available_models)}",
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data",
        help="Path to .fif file or directory containing .fif files (default: data/)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="representations",
        help="Output directory for embeddings (default: representations/)",
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)",
    )
    
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Specific layer to extract embeddings from (model-specific)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Model size variant (for Chronos, Moirai, Time-MoE)",
    )
    
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length for time series models",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        print("Available models:")
        for model in available_models:
            print(f"  - {model}")
        return 0
    
    # Check that model is provided if not listing
    if args.model is None:
        parser.error("--model is required when not using --list-models")
    
    # Build model kwargs
    model_kwargs = {}
    if args.model_size:
        model_kwargs["model_size"] = args.model_size
    if args.context_length:
        model_kwargs["context_length"] = args.context_length
    
    # Run extraction
    try:
        saved_files = extract_all(
            model_name=args.model,
            input_path=Path(args.input),
            output_dir=Path(args.output),
            device=args.device,
            layer_name=args.layer,
            verbose=args.verbose,
            **model_kwargs,
        )
        
        return 0 if len(saved_files) > 0 else 1
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

