"""Automatic cat face detection and cropping script."""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np


def crop_cat_faces(input_dir: str, output_dir: str, cascade_path: str = None):
    """
    Detect and crop cat faces from images using OpenCV cascade classifier.
    
    Args:
        input_dir: Directory containing cat images (e.g., './data/cat')
        output_dir: Directory to save cropped images
        cascade_path: Path to cat cascade classifier (uses default if None)
    """
    
    # Use default OpenCV cat cascade classifier
    if cascade_path is None:
        # OpenCV usually has this built-in, but we'll use a simple approach
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
    
    # Initialize cascade classifier
    if not os.path.exists(cascade_path):
        print(f"Cascade classifier not found at {cascade_path}")
        print("Using alternative: center crop method instead")
        return crop_center(input_dir, output_dir)
    
    cat_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Processing and saving to {output_dir}...\n")
    
    successful_crops = 0
    failed_crops = 0
    
    for image_path in tqdm(image_files, desc="Cropping cat faces"):
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                failed_crops += 1
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect cat faces
            cats = cat_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                maxSize=(500, 500)
            )
            
            if len(cats) > 0:
                # Use the largest detected face
                (x, y, w, h) = max(cats, key=lambda c: c[2] * c[3])
                
                # Add margin (20% padding)
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(img.shape[1], x + w + margin_x)
                y2 = min(img.shape[0], y + h + margin_y)
                
                # Crop image
                cropped = img[y1:y2, x1:x2]
                
                # Ensure square crop for GAN
                size = min(cropped.shape[0], cropped.shape[1])
                if size > 0:
                    cropped = cropped[:size, :size]
                    
                    # Save cropped image
                    output_path = Path(output_dir) / image_path.name
                    cv2.imwrite(str(output_path), cropped)
                    successful_crops += 1
                else:
                    failed_crops += 1
            else:
                # No face detected, use center crop as fallback
                cropped = crop_image_center(img)
                if cropped is not None:
                    output_path = Path(output_dir) / image_path.name
                    cv2.imwrite(str(output_path), cropped)
                    successful_crops += 1
                else:
                    failed_crops += 1
                    
        except Exception as e:
            failed_crops += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Cropping complete!")
    print(f"Successfully cropped: {successful_crops}")
    print(f"Failed: {failed_crops}")
    print(f"Total processed: {successful_crops + failed_crops}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


def crop_image_center(img):
    """Simple center crop for images where face detection fails."""
    if img is None or img.size == 0:
        return None
    
    h, w = img.shape[:2]
    size = min(h, w)
    
    if size < 50:  # Image too small
        return None
    
    # Crop to center square
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    
    return img[start_y:start_y + size, start_x:start_x + size]


def crop_center(input_dir: str, output_dir: str):
    """Fallback method: center crop all images to square."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Using center crop method...")
    print(f"Processing and saving to {output_dir}...\n")
    
    successful_crops = 0
    failed_crops = 0
    
    for image_path in tqdm(image_files, desc="Center cropping images"):
        try:
            img = cv2.imread(str(image_path))
            cropped = crop_image_center(img)
            
            if cropped is not None:
                output_path = Path(output_dir) / image_path.name
                cv2.imwrite(str(output_path), cropped)
                successful_crops += 1
            else:
                failed_crops += 1
                
        except Exception as e:
            failed_crops += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Center cropping complete!")
    print(f"Successfully cropped: {successful_crops}")
    print(f"Failed: {failed_crops}")
    print(f"Total processed: {successful_crops + failed_crops}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crop cat images automatically")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/cat",
        help="Input directory with cat images (default: ./data/cat)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/cat_cropped",
        help="Output directory for cropped images (default: ./data/cat_cropped)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["face", "center"],
        default="face",
        help="Cropping method: 'face' for face detection, 'center' for center crop"
    )
    
    args = parser.parse_args()
    
    if args.method == "face":
        crop_cat_faces(args.input, args.output)
    else:
        crop_center(args.input, args.output)
