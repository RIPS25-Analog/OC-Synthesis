#!/usr/bin/env python3
"""
Check annotation quality for hammer and screwdriver classes
"""

import cv2
import os
import random
from pathlib import Path

def visualize_problem_classes(img_dir, label_dir, output_dir, target_classes=[0, 2], num_samples=20):
    """
    Visualize annotations for problem classes (hammer=0, screwdriver=2)
    """
    img_path = Path(img_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    class_names = {0: 'hammer', 1: 'pliers', 2: 'screwdriver', 3: 'tape_measure', 4: 'wrench'}
    
    # Find files with target classes
    target_files = []
    
    for label_file in label_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    if class_id in target_classes:
                        target_files.append(label_file.stem)
                        break
    
    print(f"Found {len(target_files)} files with problem classes")
    
    # Sample random files
    sample_files = random.sample(target_files, min(num_samples, len(target_files)))
    
    issues_found = []
    
    for file_stem in sample_files:
        img_file = img_path / (file_stem + '.png')
        label_file = label_path / (file_stem + '.txt')
        
        if not img_file.exists():
            continue
            
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Draw annotations
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id, cx, cy, width, height = map(float, line.strip().split())
                    
                    # Convert to pixel coordinates
                    x1 = int((cx - width/2) * w)
                    y1 = int((cy - height/2) * h)
                    x2 = int((cx + width/2) * w)
                    y2 = int((cy + height/2) * h)
                    
                    # Color code by class
                    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 
                             3: (255, 255, 0), 4: (255, 0, 255)}
                    color = colors.get(class_id, (128, 128, 128))
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add class label
                    class_name = class_names.get(class_id, f'class_{int(class_id)}')
                    cv2.putText(img, class_name, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Check for potential issues
                    bbox_area = width * height
                    if class_id in target_classes:
                        if bbox_area > 0.1:  # Very large box
                            issues_found.append(f"{file_stem}: {class_name} - Very large bbox ({bbox_area:.3f})")
                        elif bbox_area < 0.005:  # Very small box
                            issues_found.append(f"{file_stem}: {class_name} - Very small bbox ({bbox_area:.3f})")
                        elif width/height > 5 or height/width > 5:  # Very elongated
                            issues_found.append(f"{file_stem}: {class_name} - Elongated bbox ({width:.3f}x{height:.3f})")
        
        # Save annotated image
        output_file = output_path / f"check_{file_stem}.png"
        cv2.imwrite(str(output_file), img)
    
    print(f"\nSaved {len(sample_files)} annotated images to {output_dir}")
    
    if issues_found:
        print(f"\n⚠️  POTENTIAL ANNOTATION ISSUES FOUND:")
        for issue in issues_found[:10]:  # Show first 10
            print(f"  → {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more issues")
    
    return issues_found

def main():
    # Update these paths
    IMG_DIR = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/YOLO11/organized_yolo_datasett/images/train"
    LABEL_DIR = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/YOLO11/organized_yolo_datasett/labels/train"
    OUTPUT_DIR = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/YOLO11/annotation_quality_check"
    
    print("Checking annotation quality for hammer and screwdriver...")
    print("=" * 60)
    
    # Check problem classes (hammer=0, screwdriver=2)
    issues = visualize_problem_classes(IMG_DIR, LABEL_DIR, OUTPUT_DIR, 
                                     target_classes=[0, 2], num_samples=20)
    
    print("\n" + "=" * 60)
    print("ANNOTATION QUALITY SUMMARY:")
    print("=" * 60)
    print(f"Files analyzed: 20 samples")
    print(f"Potential issues found: {len(issues)}")
    
    if len(issues) > 5:
        print("⚠️  HIGH number of annotation issues detected!")
        print("   → Consider reviewing and cleaning annotations")
        print("   → This could explain low performance")
    elif len(issues) > 2:
        print("⚠️  MODERATE annotation issues detected")
        print("   → Some cleanup recommended")
    else:
        print("✅ Annotations look reasonable")
        print("   → Focus on model improvements instead")
    
    print(f"\n📁 Check images in: {OUTPUT_DIR}")
    print("   → Look for bounding boxes that seem incorrect")
    print("   → Compare hammer/screwdriver with tape_measure quality")

if __name__ == "__main__":
    main()