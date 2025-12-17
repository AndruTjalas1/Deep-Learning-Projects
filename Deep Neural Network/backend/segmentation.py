"""
Image segmentation and preprocessing utilities for handwriting recognition.

Algorithm 2: Character Segmentation
Extracts individual characters from continuous handwriting using:
- Connected component analysis
- Bounding box detection
- Image preprocessing (thresholding, dilation, erosion)
"""
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import io


def preprocess_image(image_data, target_size=28):
    """
    Preprocess raw canvas data into normalized image format.
    
    Args:
        image_data: PIL Image, numpy array, or bytes
        target_size: Output image size (default 28x28 for MNIST-like)
        
    Returns:
        Preprocessed numpy array of shape (1, target_size, target_size)
    """
    # Convert to PIL Image if needed
    if isinstance(image_data, bytes):
        image_data = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, np.ndarray):
        image_data = Image.fromarray(image_data.astype('uint8'))
    
    # Convert to grayscale
    if image_data.mode != 'L':
        image_data = image_data.convert('L')
    
    # Resize to target size
    image_data = image_data.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image_data, dtype=np.float32)
    
    # Normalize to [0, 1]
    image_array = image_array / 255.0
    
    # Invert if needed (handwriting is dark on light background)
    # After normalization, ensure foreground is 1 and background is 0
    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array
    
    # Add channel dimension: (H, W) -> (1, H, W)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def segment_characters(canvas_image, min_component_size=30, max_components=50):
    """
    Segment individual characters from a canvas drawing containing continuous text.
    
    This is a key component of Algorithm 2: Character Segmentation.
    Uses connected component analysis to identify separate characters.
    
    Args:
        canvas_image: PIL Image or numpy array of the canvas
        min_component_size: Minimum pixels for a valid character
        max_components: Maximum number of characters to return
        
    Returns:
        List of dicts with character_image, bbox, and confidence
    """
    try:
        # Convert to numpy array if needed
        if isinstance(canvas_image, Image.Image):
            # Convert to RGB first, then to grayscale
            if canvas_image.mode != 'RGB':
                canvas_image = canvas_image.convert('RGB')
            canvas_array = cv2.cvtColor(np.array(canvas_image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            canvas_array = np.array(canvas_image, dtype=np.uint8)
            if len(canvas_array.shape) == 3:
                canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        # Handwriting is dark, background is light
        _, binary = cv2.threshold(canvas_array, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find connected components using scipy
        label_output = ndimage.label(binary)
        
        # Handle different scipy versions - newer versions return (array, int), older return int
        if isinstance(label_output, tuple):
            labeled_array, num_labels = label_output
        else:
            labeled_array = label_output
            num_labels = np.max(labeled_array)
        
        num_labels = int(num_labels)
        
        if num_labels == 0:
            return []
        
        # Find bounding boxes for each component
        character_segments = []
        for label_idx in range(1, num_labels + 1):
            component = (labeled_array == label_idx)
            component_size = int(np.sum(component))
            
            # Filter by size - use explicit int comparison
            if component_size < int(min_component_size):
                continue
            
            # Find bounding box
            y_coords, x_coords = np.where(component)
            if len(x_coords) == 0 or len(y_coords) == 0:
                continue
                
            x_min = int(np.min(x_coords))
            x_max = int(np.max(x_coords))
            y_min = int(np.min(y_coords))
            y_max = int(np.max(y_coords))
            
            # Extract character image with padding
            padding = 5
            x_min_padded = max(0, x_min - padding)
            x_max_padded = min(canvas_array.shape[1], x_max + padding)
            y_min_padded = max(0, y_min - padding)
            y_max_padded = min(canvas_array.shape[0], y_max + padding)
            
            # Ensure valid dimensions
            if x_max_padded <= x_min_padded or y_max_padded <= y_min_padded:
                continue
            
            char_image = canvas_array[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
            
            # Confidence is based on component fill ratio
            bbox_area = (x_max_padded - x_min_padded) * (y_max_padded - y_min_padded)
            fill_ratio = component_size / bbox_area if bbox_area > 0 else 0
            confidence = fill_ratio
            
            character_segments.append({
                'image': char_image,
                'bbox': (x_min_padded, y_min_padded, x_max_padded - x_min_padded, y_max_padded - y_min_padded),
                'confidence': confidence,
                'position_x': (x_min_padded + x_max_padded) / 2  # For left-to-right sorting
            })
        
        # Sort by horizontal position (left to right)
        character_segments.sort(key=lambda x: x['position_x'])
        
        # Limit to max_components
        character_segments = character_segments[:max_components]
        
        return character_segments
    
    except Exception as e:
        print(f"Error extracting character components: {e}")
        import traceback
        traceback.print_exc()
        return []


def standardize_character_image(char_image, target_size=28):
    """
    Standardize a character image for CNN input.
    
    Args:
        char_image: Extracted character image (numpy array)
        target_size: Target output size
        
    Returns:
        Standardized image of shape (1, target_size, target_size)
    """
    # Convert to PIL Image
    if isinstance(char_image, np.ndarray):
        # Ensure it's uint8
        if char_image.dtype != np.uint8:
            if char_image.max() <= 1.0:
                char_image = (char_image * 255).astype(np.uint8)
            else:
                char_image = char_image.astype(np.uint8)
        # Convert to PIL Image in grayscale mode
        char_image = Image.fromarray(char_image, mode='L')
    else:
        # Ensure grayscale
        if char_image.mode != 'L':
            char_image = char_image.convert('L')
    
    # Get current dimensions
    current_width, current_height = char_image.size
    
    # Calculate aspect ratio preserving resize
    aspect_ratio = current_width / current_height
    if aspect_ratio > 1:
        new_width = target_size - 4
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size - 4
        new_width = int(new_height * aspect_ratio)
    
    # Resize
    char_image = char_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create square canvas with padding
    canvas = Image.new('L', (target_size, target_size), color=255)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas.paste(char_image, (x_offset, y_offset))
    
    # Convert to numpy and normalize
    char_array = np.array(canvas, dtype=np.float32) / 255.0
    
    # Invert if needed (ensure foreground is 1, background is 0)
    if char_array.mean() > 0.5:
        char_array = 1.0 - char_array
    
    # Add channel dimension
    char_array = np.expand_dims(char_array, axis=0)
    
    return char_array


def batch_segment_and_standardize(canvas_image, target_size=28):
    """
    Segment all characters and standardize them for batch processing.
    
    Args:
        canvas_image: Canvas image containing handwritten text
        target_size: Target image size for CNN
        
    Returns:
        np.ndarray of shape (num_chars, 1, target_size, target_size)
        List of bounding boxes
    """
    try:
        segments = segment_characters(canvas_image)
        
        standardized_images = []
        bounding_boxes = []
        
        for segment in segments:
            try:
                std_image = standardize_character_image(segment['image'], target_size)
                # Verify shape
                if std_image.shape != (1, target_size, target_size):
                    print(f"Warning: unexpected shape {std_image.shape}, expected (1, {target_size}, {target_size})")
                standardized_images.append(std_image)
                bounding_boxes.append(segment['bbox'])
            except Exception as e:
                print(f"Error standardizing character segment: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if standardized_images:
            # Stack along a new axis to preserve shape (num_images, 1, height, width)
            batch = np.stack(standardized_images, axis=0)
            print(f"Batch shape after stacking: {batch.shape}")
        else:
            batch = np.array([]).reshape(0, 1, target_size, target_size)
        
        return batch, bounding_boxes
    
    except Exception as e:
        print(f"Error in batch_segment_and_standardize: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]).reshape(0, 1, target_size, target_size), []
