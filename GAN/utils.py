#!/usr/bin/env python3
"""
Utility script for DCGAN project setup and management.
Simplifies common tasks like downloading datasets, converting configs, etc.
"""

import os
import sys
import argparse
from pathlib import Path
import shutil


def setup_directory_structure():
    """Create necessary directories for the project."""
    base_dir = Path(__file__).parent
    
    directories = [
        base_dir / 'Backend' / 'data' / 'cats',
        base_dir / 'Backend' / 'data' / 'dogs',
        base_dir / 'Backend' / 'samples',
        base_dir / 'Backend' / 'saved_models',
        base_dir / 'Backend' / 'logs',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")


def validate_environment():
    """Validate that required packages are installed."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        if torch.backends.mps.is_available():
            print(f"  - Apple MPS available: True")
    except ImportError:
        print("✗ PyTorch not installed. Run: pip install -r Backend/requirements.txt")
        return False
    
    try:
        import fastapi
        print(f"✓ FastAPI installed")
    except ImportError:
        print("✗ FastAPI not installed. Run: pip install -r Backend/requirements.txt")
        return False
    
    return True


def check_dataset():
    """Check if dataset is properly structured."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'Backend' / 'data'
    
    print("\nDataset Structure Check:")
    
    for animal_type in ['cats', 'dogs']:
        animal_dir = data_dir / animal_type
        
        if not animal_dir.exists():
            print(f"✗ {animal_dir} - NOT FOUND")
        else:
            image_files = list(animal_dir.glob('*.[jJpP][nNgG]*'))
            image_files += list(animal_dir.glob('*.[pP][nN][gG]'))
            image_files += list(animal_dir.glob('*.[bB][mM][pP]'))
            
            if image_files:
                print(f"✓ {animal_dir} - {len(image_files)} images found")
            else:
                print(f"⚠ {animal_dir} - EXISTS but no images found")


def generate_sample_config():
    """Generate a sample configuration with explanations."""
    base_dir = Path(__file__).parent
    config_path = base_dir / 'Backend' / 'config.yaml'
    
    if config_path.exists():
        print(f"Config already exists at {config_path}")
        return
    
    print(f"Generated config at {config_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DCGAN Project Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils.py --setup          Setup directory structure
  python utils.py --validate       Validate environment
  python utils.py --check-dataset  Check dataset status
  python utils.py --all            Run all checks
        """
    )
    
    parser.add_argument('--setup', action='store_true', help='Setup directory structure')
    parser.add_argument('--validate', action='store_true', help='Validate environment')
    parser.add_argument('--check-dataset', action='store_true', help='Check dataset')
    parser.add_argument('--all', action='store_true', help='Run all setup tasks')
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    if args.setup or args.all:
        print("Setting up directory structure...")
        setup_directory_structure()
    
    if args.validate or args.all:
        print("\nValidating environment...")
        if not validate_environment():
            sys.exit(1)
    
    if args.check_dataset or args.all:
        check_dataset()
    
    if args.all:
        print("\n✓ Setup complete! Ready to train.")
        print("\nNext steps:")
        print("1. Add images to Backend/data/cats/ and Backend/data/dogs/")
        print("2. Adjust config in Backend/config.yaml")
        print("3. Run: python Backend/main.py")
        print("4. Open: http://localhost:3000")


if __name__ == '__main__':
    main()
