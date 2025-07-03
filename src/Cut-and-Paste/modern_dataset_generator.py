import argparse
import glob
import os
import json
import yaml
import cv2
import numpy as np
import random
from PIL import Image, ImageFilter
from multiprocessing import Pool
from functools import partial
import signal
from collections import namedtuple
from typing import List, Tuple, Dict, Optional
import logging
import shutil
from time import time
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

class ModernDatasetGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.min_objects = config.get('min_objects', 1)
        self.max_objects = config.get('max_objects', 3)
        self.min_scale = config.get('min_scale', 0.3)
        self.max_scale = config.get('max_scale', 1.0)
        self.max_rotation = config.get('max_rotation', 30)
        self.max_iou = config.get('max_iou', 0.3)
        self.blur_probability = config.get('blur_probability', 0.3)
        self.num_workers = config.get('num_workers', 8)

        self.train_ratio = config.get('train_ratio', 0.7)
        self.val_ratio = config.get('val_ratio', 0.2)
        
    def get_bounding_box_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Extract bounding box coordinates from a binary mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return -1, -1, -1, -1
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return int(xmin), int(ymin), int(xmax), int(ymax)
    
    def get_iou(self, box1: Rectangle, box2: Rectangle) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        dx = min(box1.xmax, box2.xmax) - max(box1.xmin, box2.xmin)
        dy = min(box1.ymax, box2.ymax) - max(box1.ymin, box2.ymin)
        
        if dx <= 0 or dy <= 0:
            return 0.0
            
        intersection = dx * dy
        area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
        area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def normalize_object_size(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Normalize object size relative to output image dimensions."""
        # Calculate scaling factor to normalize object size
        img_width, img_height = image.size
        scale_x = self.width / img_width
        scale_y = self.height / img_height
        scale_factor = min(scale_x, scale_y)*0.9  # Don't upscale if already small
        
        # if scale_factor >= 1.0:
        #     return image, mask  # No need to resize if already fits within output size
        
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        if new_width <= 1 or new_height <=1:
            return image, mask  # Avoid resizing to zero/single pixel width or height
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        mask = mask.resize((new_width, new_height), Image.NEAREST)
        
        assert image.size[0] <= self.width and image.size[1] <= self.height, \
            f"Object size {image.size} exceeds output size {self.width}x{self.height}"
        
        return image, mask
    
    def apply_augmentations(self, obj_width, obj_height, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply random augmentations to image and mask."""
        # Scale augmentation (applied after normalization)
        if random.random() < 0.7:  # 70% chance to apply scaling
            scale = random.uniform(self.min_scale, self.max_scale)
            scale = min(scale, self.width / obj_width * 0.9, self.height / obj_height * 0.9)  # Object can't be bigger than image
            new_size = (int(image.width * scale), int(image.height * scale))

            # Object should be atleast 5 pixels in width and height after scaling
            if obj_width*scale > 4 and obj_height*scale > 4:
                image = image.resize(new_size, Image.LANCZOS)
                mask = mask.resize(new_size, Image.NEAREST)
        
        # Rotation augmentation
        if random.random() < 0.5:  # 50% chance to apply rotation
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            image = image.rotate(angle, expand=True, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, expand=True, fillcolor=0)
        
        # Motion blur simulation
        if random.random() < self.blur_probability:
            blur_radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return image, mask
    
    def synthesize_image(self, background_path: str, objects: List[Tuple[str, str]], 
                        output_image_path: str, output_annotation_path: str) -> bool:
        """Synthesize a single image with multiple objects."""
        if True:
            # Load background
            background = Image.open(background_path).convert('RGB')
            background = background.resize((self.width, self.height), Image.LANCZOS)
            
            annotations = []
            placed_boxes = []
            
            for obj_path, label in objects:
                # Load object image and mask
                assert os.path.exists(obj_path), f"Object image not found: {obj_path}"
                
                mask_path = obj_path.replace('.jpg', '_mask.jpg')
                assert os.path.exists(mask_path), f"Mask not found: {mask_path}"
                
                obj_img = Image.open(obj_path).convert('RGB')
                mask_img = Image.open(mask_path).convert('L')
                
                # First normalize object size relative to output image
                obj_img, mask_img = self.normalize_object_size(obj_img, mask_img)

                # Get object bounding box from mask
                xmin, ymin, xmax, ymax = self.get_bounding_box_from_mask(np.array(mask_img) > 128)

                if xmin == -1:  # Invalid mask
                    continue

                obj_width, obj_height = xmax - xmin, ymax - ymin

                assert obj_width > 1 and obj_height > 1, f"Before augmentation: Invalid object size: {obj_width}x{obj_height} for {label}"
                assert obj_width <= self.width and obj_height <= self.height, \
                    f"Before augmentation: Object size {obj_width}x{obj_height} exceeds image size {self.width}x{self.height} for {label}"
                
                # Then apply augmentations
                obj_img, mask_img = self.apply_augmentations(obj_width, obj_height, obj_img, mask_img)

                # Get object bounding box from mask
                xmin, ymin, xmax, ymax = self.get_bounding_box_from_mask(np.array(mask_img) > 128)

                if xmin == -1:  # Invalid mask
                    continue

                obj_width, obj_height = xmax - xmin, ymax - ymin

                assert obj_width > 1 and obj_height > 1, f"After augmentation: Invalid object size: {obj_width}x{obj_height} for {label}"
                assert obj_width <= self.width and obj_height <= self.height, \
                    f"After augmentation: Object size {obj_width}x{obj_height} exceeds image size {self.width}x{self.height} for {label}"

                # Find valid placement position
                max_attempts = 50
                placed = False
                
                for _ in range(max_attempts):
                    # Random position within image bounds
                    new_x_center = random.randint(obj_width//2, self.width - obj_width//2)
                    new_y_center = random.randint(obj_height//2, self.height - obj_height//2)

                    new_box = Rectangle(new_x_center - obj_width//2, new_y_center - obj_height//2,
                                        new_x_center + obj_width//2, new_y_center + obj_height//2)

                    # Check for overlaps with previously placed objects
                    if any(self.get_iou(new_box, placed_box) > self.max_iou for placed_box in placed_boxes):
                        continue
                    
                    # Restrict new box to image bounds
                    new_box = Rectangle(
                        max(0, new_box.xmin), max(0, new_box.ymin),
                        min(self.width, new_box.xmax), min(self.height, new_box.ymax)
                    )

                    # Ensure atleast 50% of the object is within the image bounds
                    new_box_area = (new_box.xmax - new_box.xmin) * (new_box.ymax - new_box.ymin)
                    obj_area = obj_width * obj_height
                    if new_box_area / obj_area < 0.5:
                        # logger.info(f"Skipped object because only {new_box_area/obj_area:.2f} of it is within the image bounds: {new_box}")
                        continue

                    # Paste object onto background
                    background.paste(obj_img, (new_box.xmin, new_box.ymin), mask_img)

                    # Record annotation
                    annotations.append({
                        'label': label,
                        'bbox': list(new_box)
                    })
                    placed_boxes.append(new_box)
                    placed = True
                    break
            
                if not placed:
                    logger.warning(f"Could not place object {label} in image")
            
            if not annotations:
                return False
            
            # Save synthesized image
            background.save(output_image_path, quality=95)
            
            # Save annotations in YOLO format
            self.save_yolo_annotation(annotations, output_annotation_path)
            
            return True
            
        # except Exception as e:
        #     logger.error(f"Error synthesizing image: {e}")
        #     return False
    
    def save_yolo_annotation(self, annotations: List[Dict], output_path: str):
        """Save annotations in YOLO format."""
        with open(output_path, 'w') as f:
            for ann in annotations:
                label_id = self.label_to_id.get(ann['label'], 0)
                bbox = ann['bbox']
                
                # Convert to YOLO format (normalized center_x, center_y, width, height)
                center_x = (bbox[0] + bbox[2]) / 2.0 / self.width
                center_y = (bbox[1] + bbox[3]) / 2.0 / self.height
                width = (bbox[2] - bbox[0]) / self.width
                height = (bbox[3] - bbox[1]) / self.height

                assert 0 <= width <= 1 and 0 <= height <= 1, f"Normalized dimensions out of bounds: {width}, {height}"
                assert 0 <= center_x <= 1 and 0 <= center_y <= 1, f"Normalized coordinates out of bounds: {center_x}, {center_y}"

                f.write(f"{label_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def collect_object_files(self, root_dir: str) -> Tuple[List[str], List[str]]:
        """Collect object images and their labels from directory structure."""
        object_files = []
        labels = []
        
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for img_file in glob.glob(os.path.join(class_path, 'images', '*.jpg')) + \
                           glob.glob(os.path.join(class_path, 'images', '*.png')):
                if 'mask' in img_file:
                    continue
                object_files.append(img_file)
                labels.append(class_dir)
        
        return object_files, labels
    
    def create_label_mapping(self, labels: List[str]) -> Dict[str, int]:
        """Create mapping from label names to IDs."""
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def split_dataset(self, output_dir: str, num_images: int) -> Dict[str, List[int]]:
        """Split dataset indices into train/val/test sets."""
        # Set random seed for reproducible splits
        np.random.seed(0)
        
        # Create shuffled indices
        indices = np.arange(num_images)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(num_images * self.train_ratio)
        val_size = int(num_images * self.val_ratio)
        test_size = num_images - train_size - val_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        logger.info(f"Dataset split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        return {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist()
        }
    
    def create_split_directories(self, output_dir: str):
        """Create train/val/test directory structure."""
        for split in ['train', 'val', 'test']:
            split_images_dir = os.path.join(output_dir, split, 'images')
            split_labels_dir = os.path.join(output_dir, split, 'labels')
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)
    
    def move_files_to_splits(self, output_dir: str, split_mapping: Dict[str, List[int]]):
        """Move generated files to appropriate train/val/test directories."""
        temp_images_dir = os.path.join(output_dir, 'images')
        temp_labels_dir = os.path.join(output_dir, 'labels')
        
        for split, indices in split_mapping.items():
            split_images_dir = os.path.join(output_dir, split, 'images')
            split_labels_dir = os.path.join(output_dir, split, 'labels')
            
            for idx in indices:
                # Source files
                src_image = os.path.join(temp_images_dir, f'synthetic_{idx:06d}.jpg')
                src_label = os.path.join(temp_labels_dir, f'synthetic_{idx:06d}.txt')
                
                # Destination files
                dst_image = os.path.join(split_images_dir, f'synthetic_{idx:06d}.jpg')
                dst_label = os.path.join(split_labels_dir, f'synthetic_{idx:06d}.txt')
                
                # Move files if they exist
                if os.path.exists(src_image):
                    shutil.move(src_image, dst_image)
                if os.path.exists(src_label):
                    shutil.move(src_label, dst_label)
        
        # Remove temporary directories if empty
        try:
            if os.path.exists(temp_images_dir) and not os.listdir(temp_images_dir):
                os.rmdir(temp_images_dir)
            if os.path.exists(temp_labels_dir) and not os.listdir(temp_labels_dir):
                os.rmdir(temp_labels_dir)
        except OSError:
            pass  # Directories not empty
    
    def generate_dataset(self, objects_dir: str, backgrounds_dir: str, 
                        output_dir: str, num_images: int):
        random.seed(0)
        
        """Generate the complete synthetic dataset."""        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.create_split_directories(output_dir)
        
        # Also create temporary directories for initial generation
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Collect object files and labels
        object_files, object_labels = self.collect_object_files(objects_dir)
        if not object_files:
            raise ValueError(f"No object files found in {objects_dir}")
        
        # Create label mapping
        self.label_to_id = self.create_label_mapping(object_labels)
        unique_labels = list(self.label_to_id.keys())
        
        # Collect background files
        background_files = glob.glob(os.path.join(backgrounds_dir, '*.jpg')) + \
                          glob.glob(os.path.join(backgrounds_dir, '*.png'))
        if not background_files:
            raise ValueError(f"No background files found in {backgrounds_dir}")
        
        logger.info(f"Found {len(object_files)} object files with {len(unique_labels)} classes")
        logger.info(f"Unique labels: {unique_labels}")
        logger.info(f"Found {len(background_files)} background files")
        
        # Generate synthesis parameters
        synthesis_params = []
        for i in range(num_images):
            # Select random background
            bg_file = random.choice(background_files)
            
            # Select random objects
            num_objects = random.randint(self.min_objects, self.max_objects)
            selected_objects = []
            
            for _ in range(num_objects):
                idx = random.randint(0, len(object_files) - 1)
                selected_objects.append((object_files[idx], object_labels[idx]))
            
            output_img = os.path.join(images_dir, f'synthetic_{i:06d}.jpg')
            output_ann = os.path.join(labels_dir, f'synthetic_{i:06d}.txt')
            
            synthesis_params.append((bg_file, selected_objects, output_img, output_ann))
        
        start_time = time()
        
        # Generate images using multiprocessing
        logger.info(f"Generating {num_images} synthetic images...")
        partial_func = partial(self._synthesize_wrapper)
        
        with Pool(self.num_workers, init_worker) as pool:
            try:
                results = pool.map(partial_func, synthesis_params)
                successful = sum(results)
                logger.info(f"Successfully generated {successful}/{num_images} images")
            except KeyboardInterrupt:
                logger.info("Generation interrupted by user")
                pool.terminate()
                pool.join()
                return

        # Split dataset into train/val/test
        split_mapping = self.split_dataset(output_dir, num_images)
        self.move_files_to_splits(output_dir, split_mapping)
        
        # Generate YAML config file
        self.generate_yaml_config(os.path.join(output_dir), unique_labels)
        
        logger.info(f"Dataset generation took {time() - start_time:.2f} seconds")
        logger.info(f"Dataset generation complete. Output saved to {output_dir}")
    
    def _synthesize_wrapper(self, params):
        """Wrapper for multiprocessing."""
        bg_file, objects, output_img, output_ann = params
        return self.synthesize_image(bg_file, objects, output_img, output_ann)
    
    def generate_yaml_config(self, output_dir: str, class_names: List[str]):
        """Generate YAML configuration file for the dataset."""
        config = {
            'nc': len(class_names),
            'names': class_names,
            'train': os.path.join(output_dir, 'train', 'images'),
            'val': os.path.join(output_dir, 'val', 'images'),
            'test': os.path.join(output_dir, 'test', 'images')
        }
        
        dataset_name = output_dir.split(os.path.sep)[-1]
        yaml_path = os.path.join(output_dir, '..', '..', f'{dataset_name}.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Dataset config saved to {yaml_path}")

def init_worker():
    """Initialize worker process to handle interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    parser = argparse.ArgumentParser(description="Modern synthetic dataset generator")
    parser.add_argument("--objects_dir", help="Directory containing object images organized by class")
    parser.add_argument("--backgrounds_dir", help="Directory containing background images")
    parser.add_argument("--output_dir", help="Output directory for synthetic dataset")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--config", help="JSON config file with generation parameters")
    parser.add_argument("--width", type=int, default=640, help="Output image width")
    parser.add_argument("--height", type=int, default=480, help="Output image height")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'width': args.width,
            'height': args.height,
            'num_workers': args.workers,
            'min_objects': 1,
            'max_objects': 3,
            'min_scale': 0.3,
            'max_scale': 1.0,
            'max_rotation': 30,
            'max_iou': 0.3,
            'blur_probability': 0.3,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio
        }
    
    # Generate dataset
    generator = ModernDatasetGenerator(config)
    generator.generate_dataset(
        args.objects_dir,
        args.backgrounds_dir,
        args.output_dir,
        args.num_images
    )

if __name__ == '__main__':
    main()
