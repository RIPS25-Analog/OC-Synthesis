#!/usr/bin/env python3
"""
FIXED Dataset preparation script with correct class mapping
This will properly handle all 5 classes and create correct YOLO annotations
"""

import os
import shutil
import json
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

def analyze_original_data(source_dir):
    """
    Analyze the original dataset to understand class mapping issues
    """
    source_path = Path(source_dir)
    
    print("🔍 Analyzing original dataset structure...")
    
    # Check required directories
    required_dirs = ['compositional_image', 'yolo_annotations', 'label']
    for dir_name in required_dirs:
        dir_path = source_path / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
        print(f"✅ Found: {dir_name}/ ({len(list(dir_path.glob('*')))} files)")
    
    # Analyze JSON labels
    label_dir = source_path / 'label'
    json_classes = Counter()
    file_to_class = {}
    
    print("\n📊 Analyzing JSON labels...")
    for json_file in label_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'class' in data:
                    class_name = data['class']
                    json_classes[class_name] += 1
                    file_to_class[json_file.stem] = class_name
        except Exception as e:
            print(f"Warning: Error reading {json_file}: {e}")
    
    print("Classes found in JSON:")
    for class_name, count in sorted(json_classes.items()):
        print(f"   {class_name}: {count} instances")
    
    # Analyze YOLO annotations  
    yolo_dir = source_path / 'yolo_annotations'
    yolo_classes = Counter()
    file_to_yolo_class = {}
    
    print("\n📊 Analyzing YOLO annotations...")
    for yolo_file in yolo_dir.glob('*.txt'):
        try:
            with open(yolo_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    yolo_classes[class_id] += 1
                    file_to_yolo_class[yolo_file.stem] = class_id
        except Exception as e:
            print(f"Warning: Error reading {yolo_file}: {e}")
    
    print("Class IDs found in YOLO:")
    for class_id, count in sorted(yolo_classes.items()):
        print(f"   Class {class_id}: {count} instances")
    
    return file_to_class, file_to_yolo_class, sorted(json_classes.keys())

def create_correct_mapping(file_to_class, file_to_yolo_class, all_classes):
    """
    Create the correct mapping from JSON class names to YOLO class IDs
    """
    print("\n🎯 Creating correct class mapping...")
    
    # Create alphabetical mapping
    class_name_to_id = {class_name: i for i, class_name in enumerate(sorted(all_classes))}
    
    print("Correct mapping should be:")
    for class_name, class_id in class_name_to_id.items():
        print(f"   {class_id}: {class_name}")
    
    # Analyze what the current YOLO files are doing wrong
    print("\n🔍 Current YOLO mapping issues:")
    current_mapping = defaultdict(set)
    
    for file_stem in file_to_class.keys():
        if file_stem in file_to_yolo_class:
            json_class = file_to_class[file_stem]
            yolo_class = file_to_yolo_class[file_stem]
            current_mapping[yolo_class].add(json_class)
    
    for yolo_id, json_classes in current_mapping.items():
        print(f"   YOLO ID {yolo_id} → {json_classes}")
    
    return class_name_to_id

def prepare_corrected_dataset(source_dir, output_dir, class_mapping):
    """
    Prepare the dataset with corrected class mapping
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    print(f"\n📁 Creating output directory: {output_path}")
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all valid files
    image_dir = source_path / 'compositional_image'
    yolo_dir = source_path / 'yolo_annotations'
    label_dir = source_path / 'label'
    
    valid_files = []
    file_to_class = {}
    
    for img_file in image_dir.glob('*.png'):
        file_stem = img_file.stem
        yolo_file = yolo_dir / (file_stem + '.txt')
        json_file = label_dir / (file_stem + '.json')
        
        if yolo_file.exists() and json_file.exists():
            try:
                # Get class from JSON
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    class_name = data.get('class')
                    if class_name and class_name in class_mapping:
                        valid_files.append(file_stem)
                        file_to_class[file_stem] = class_name
            except:
                pass
    
    print(f"\n📋 Found {len(valid_files)} valid files to process")
    
    # Split the data
    train_files, temp_files = train_test_split(valid_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    splits = {
        'train': train_files, 
        'val': val_files, 
        'test': test_files
    }
    
    print(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Process each split
    for split_name, file_list in splits.items():
        print(f"\n🔄 Processing {split_name} split...")
        
        class_counts = Counter()
        
        for i, file_stem in enumerate(file_list):
            if i % 200 == 0 and i > 0:
                print(f"   Processed {i}/{len(file_list)} files...")
            
            # Copy image
            src_img = image_dir / (file_stem + '.png')
            dst_img = output_path / 'images' / split_name / (file_stem + '.png')
            shutil.copy2(src_img, dst_img)
            
            # Create corrected annotation
            class_name = file_to_class[file_stem]
            correct_class_id = class_mapping[class_name]
            class_counts[class_name] += 1
            
            # Read original YOLO annotation format
            src_yolo = yolo_dir / (file_stem + '.txt')
            dst_yolo = output_path / 'labels' / split_name / (file_stem + '.txt')
            
            with open(src_yolo, 'r') as f:
                lines = f.readlines()
            
            # Write corrected annotation
            with open(dst_yolo, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        # Replace class ID with correct one
                        parts[0] = str(correct_class_id)
                        f.write(' '.join(parts) + '\n')
        
        print(f"   {split_name} class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"     {class_name}: {count}")
    
    return output_path

def create_dataset_yaml(output_dir, class_mapping):
    """
    Create the corrected dataset.yaml file
    """
    output_path = Path(output_dir)
    
    yaml_content = f"""# CORRECTED Dataset configuration for YOLOv11
# Generated with proper class mapping

path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(class_mapping)}

# Class names (corrected mapping)
names:
"""
    
    # Sort by class ID
    sorted_mapping = sorted(class_mapping.items(), key=lambda x: x[1])
    for class_name, class_id in sorted_mapping:
        yaml_content += f"  {class_id}: {class_name}\n"
    
    yaml_file = output_path / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n📄 Created dataset.yaml at: {yaml_file}")
    return yaml_file

def main():
    # ========================================
    # UPDATE THESE PATHS FOR YOUR SYSTEM
    # ========================================
    
    SOURCE_DIR = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval/insertion_ilog2_istren2_context_surfrandom_cov15-50_VISIBILITY"
    OUTPUT_DIR = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/YOLO11/organized_yolo_datasett"
    
    print("🛠️  YOLO DATASET REGENERATION - PROPER FIX")
    print("=" * 60)
    
    try:
        # Step 1: Analyze original data
        file_to_class, file_to_yolo_class, all_classes = analyze_original_data(SOURCE_DIR)
        
        # Step 2: Create correct mapping
        class_mapping = create_correct_mapping(file_to_class, file_to_yolo_class, all_classes)
        
        # Step 3: Prepare corrected dataset
        dataset_path = prepare_corrected_dataset(SOURCE_DIR, OUTPUT_DIR, class_mapping)
        
        # Step 4: Create dataset.yaml
        yaml_file = create_dataset_yaml(OUTPUT_DIR, class_mapping)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Dataset regenerated with proper class mapping")
        print("=" * 60)
        print(f"📁 Dataset location: {dataset_path}")
        print(f"📄 Config file: {yaml_file}")
        print(f"\n🎯 Found {len(class_mapping)} classes:")
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            print(f"   {class_id}: {class_name}")
        
        print(f"\n🚀 Next step - Train with:")
        print(f"   python yolo_finetune.py --data {yaml_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. SOURCE_DIR path is correct")
        print("2. Required folders exist (compositional_image, yolo_annotations, label)")
        print("3. You have write permissions to OUTPUT_DIR")

if __name__ == "__main__":
    main()