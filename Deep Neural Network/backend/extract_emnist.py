import gzip
import os
from pathlib import Path

data_dir = Path("emnist_data")
data_dir.mkdir(exist_ok=True)

# Find all .gz files
gz_files = list(data_dir.glob("*.gz"))

if not gz_files:
    print("No .gz files found in emnist_data/")
    print("Please download EMNIST from: https://www.nist.gov/itl/products-and-services/emnist-dataset")
    print("Download 'Binary format as the original MNIST dataset' and extract to emnist_data/")
else:
    for gz_file in gz_files:
        out_file = gz_file.with_suffix('')
        print(f"Extracting {gz_file.name}...")
        
        with gzip.open(gz_file, 'rb') as f_in:
            with open(out_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"  -> {out_file.name}")
    
    print("\nAll files extracted successfully!")
    
    # Verify extracted files
    expected = [
        "emnist-balanced-train-images-idx3-ubyte",
        "emnist-balanced-train-labels-idx1-ubyte",
        "emnist-balanced-test-images-idx3-ubyte",
        "emnist-balanced-test-labels-idx1-ubyte"
    ]
    
    found = [f for f in expected if (data_dir / f).exists()]
    print(f"\nFound {len(found)}/{len(expected)} required files:")
    for f in found:
        size = (data_dir / f).stat().st_size / (1024*1024)
        print(f"  [OK] {f} ({size:.1f} MB)")
