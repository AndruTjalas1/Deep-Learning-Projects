"""Automatic dog face detection and cropping script."""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np


def crop_dog_faces(input_dir: str, output_dir: str, cascade_path: str = None):
    """
    Detect and crop dog faces from images using OpenCV cascade classifier.
    
    Args:
        input_dir: Directory containing dog images (e.g., './data/dog')
        output_dir: Directory to save cropped images
        cascade_path: Path to dog cascade classifier (uses default if None)
    """
    
    # Use default OpenCV dog cascade classifier
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Initialize cascade classifier
    if not os.path.exists(cascade_path):
        print(f"Cascade classifier not found at {cascade_path}")
        print("Using alternative: center crop method instead")
        return crop_center(input_dir, output_dir)
    
    dog_cascade = cv2.CascadeClassifier(cascade_path)
    
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
    
    for image_path in tqdm(image_files, desc="Cropping dog faces"):
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                failed_crops += 1
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect dog faces
            dogs = dog_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                maxSize=(500, 500)
            )
            
            if len(dogs) > 0:
                # Use the largest detected face
                (x, y, w, h) = max(dogs, key=lambda c: c[2] * c[3])
                
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
            print(f"Error processing {image_path}: {e}")
            failed_crops += 1
    
    print(f"\n{'='*50}")
    print(f"Cropping complete!")
    print(f"Successful: {successful_crops}")
    print(f"Failed: {failed_crops}")
    print(f"{'='*50}")


def crop_image_center(img, target_size=None):
    """
    Center crop an image to a square.
    
    Args:
        img: Input image
        target_size: Target size (if None, uses the smaller dimension)
    
    Returns:
        Cropped square image or None if crop fails
    """
    h, w = img.shape[:2]
    size = min(h, w)
    
    if size <= 0:
        return None
    
    top = (h - size) // 2
    left = (w - size) // 2
    
    return img[top:top+size, left:left+size]


def crop_center(input_dir: str, output_dir: str):
    """
    Center crop all images when face detection fails.
    
    Args:
        input_dir: Directory containing dog images
        output_dir: Directory to save cropped images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Using center crop method...\n")
    
    successful_crops = 0
    failed_crops = 0
    
    for image_path in tqdm(image_files, desc="Center cropping"):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                failed_crops += 1
                continue
            
            cropped = crop_image_center(img)
            if cropped is not None:
                output_path = Path(output_dir) / image_path.name
                cv2.imwrite(str(output_path), cropped)
                successful_crops += 1
            else:
                failed_crops += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            failed_crops += 1
    
    print(f"\n{'='*50}")
    print(f"Center cropping complete!")
    print(f"Successful: {successful_crops}")
    print(f"Failed: {failed_crops}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crop dog faces from images")
    parser.add_argument("--input", type=str, default="./data/dog",
                        help="Input directory containing dog images")
    parser.add_argument("--output", type=str, default="./data/dog_cropped",
                        help="Output directory for cropped images")
    parser.add_argument("--cascade", type=str, default=None,
                        help="Path to cascade classifier (optional)")
    
    args = parser.parse_args()
    
    crop_dog_faces(args.input, args.output, args.cascade)
