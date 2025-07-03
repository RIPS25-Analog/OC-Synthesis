import yaml
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
from PIL import Image
import cv2

class DatasetStatsChecker:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        
        with open(self.yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dataset_name = Path(yaml_path).stem
        self.output_dir = Path(f"runs/{self.dataset_name}/dataset-stats")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics storage
        self.stats = {
            'total_images': {'train': 0, 'val': 0, 'test': 0},
            'total_objects': {'train': 0, 'val': 0, 'test': 0},
            'class_counts': {'train': Counter(), 'val': Counter(), 'test': Counter()},
            'objects_per_image': {'train': [], 'val': [], 'test': []},
            'bbox_centers': {'train': [], 'val': [], 'test': []},
            'bbox_sizes': {'train': [], 'val': [], 'test': []}
        }
    
    def analyze_split(self, split_name, images_dir):
        """Analyze a single data split (train/val/test)"""
        print(f"Analyzing {split_name} split...")
        
        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} does not exist")
            return
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(images_dir).glob(ext))
        
        self.stats['total_images'][split_name] = len(image_files)
        
        for img_path in image_files:
            # Load annotations
            annotation_path = str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
            
            assert os.path.exists(annotation_path), f"Error: Annotation file {annotation_path} not found for image {img_path}"

            annotations = []
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 5, f"Error: Invalid annotation format in {annotation_path} for image {img_path}"
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
            
            objects_in_image = len(annotations)
            self.stats['objects_per_image'][split_name].append(objects_in_image)
            self.stats['total_objects'][split_name] += objects_in_image
            
            for class_id, x_center, y_center, bbox_width, bbox_height in annotations:
                self.stats['class_counts'][split_name][class_id] += 1
                self.stats['bbox_centers'][split_name].append((x_center, y_center))
                self.stats['bbox_sizes'][split_name].append((bbox_width, bbox_height))
    
    def analyze_dataset(self):
        """Analyze the entire dataset"""
        print(f"Analyzing dataset: {self.dataset_name}")
        
        # Analyze each split
        base_path = Path(self.yaml_path).parent
        
        for split in ['train', 'val', 'test']:
            if split in self.config:
                images_dir = base_path / self.config[split]
                self.analyze_split(split, images_dir)
    
    def plot_bbox_center_distribution(self):
        """Create heatmap of bounding box center positions"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        splits = ['train', 'val', 'test']
        
        for i, split in enumerate(splits):
            centers = self.stats['bbox_centers'][split]
            if not centers:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[i].set_title(f'{split.capitalize()} - Bbox Centers')
                continue
            
            x_coords = [c[0] for c in centers]
            y_coords = [c[1] for c in centers]
            
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50, range=[[0, 1], [0, 1]])
            
            im = axes[i].imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', interpolation='gaussian')
            axes[i].set_xlabel('X Center (normalized)')
            axes[i].set_ylabel('Y Center (normalized)')
            axes[i].set_title(f'{split.capitalize()} - Bbox Centers (n={len(centers)})')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bbox_center_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distribution(self):
        """Create histogram of class distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        class_names = self.config.get('names', [f'Class {i}' for i in range(self.config['nc'])])
        splits = ['train', 'val', 'test']
        
        # Individual split histograms
        for i, split in enumerate(splits):
            class_counts = self.stats['class_counts'][split]
            
            if not class_counts:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[i].set_title(f'{split.capitalize()} Class Distribution')
                continue
            
            classes = list(range(len(class_names)))
            counts = [class_counts.get(c, 0) for c in classes]
            
            bars = axes[i].bar(classes, counts, color=plt.cm.tab10(np.linspace(0, 1, len(classes))))
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{split.capitalize()} Class Distribution')
            axes[i].set_xticks(classes)
            axes[i].set_xticklabels(class_names, rotation=45, ha='right')
        
        # Combined histogram
        combined_counts = Counter()
        for split in splits:
            for class_id, count in self.stats['class_counts'][split].items():
                combined_counts[class_id] += count
        
        classes = list(range(len(class_names)))
        counts = [combined_counts.get(c, 0) for c in classes]
        
        bars = axes[3].bar(classes, counts, color=plt.cm.tab10(np.linspace(0, 1, len(classes))))
        axes[3].set_xlabel('Class')
        axes[3].set_ylabel('Count')
        axes[3].set_title('Combined Class Distribution')
        axes[3].set_xticks(classes)
        axes[3].set_xticklabels(class_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_objects_per_image(self):
        """Create histogram of objects per image"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        splits = ['train', 'val', 'test']
        
        # Individual split histograms
        for i, split in enumerate(splits):
            objects_per_img = self.stats['objects_per_image'][split]
            
            if not objects_per_img:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[i].set_title(f'{split.capitalize()} Objects per Image')
                continue
            
            # Use integer bins for objects per image
            max_objects = max(objects_per_img)
            bins = np.arange(0, max_objects + 2) - 0.5  # Centered on integers
            
            axes[i].hist(objects_per_img, bins=bins, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel('Number of Objects')
            axes[i].set_ylabel('Number of Images')
            axes[i].set_title(f'{split.capitalize()} Objects per Image (avg: {np.mean(objects_per_img):.1f})')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(range(0, max_objects + 1))
        
        # Combined histogram
        all_objects = []
        for split in splits:
            all_objects.extend(self.stats['objects_per_image'][split])
        
        if all_objects:
            max_objects = max(all_objects)
            bins = np.arange(0, max_objects + 2) - 0.5  # Centered on integers
            
            axes[3].hist(all_objects, bins=bins, alpha=0.7, edgecolor='black')
            axes[3].set_xlabel('Number of Objects')
            axes[3].set_ylabel('Number of Images')
            axes[3].set_title(f'Combined Objects per Image (avg: {np.mean(all_objects):.1f})')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xticks(range(0, max_objects + 1))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'objects_per_image.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bbox_size_distribution(self):
        """Create scatter plot of bounding box sizes"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        splits = ['train', 'val', 'test']
        
        for i, split in enumerate(splits):
            bbox_sizes = self.stats['bbox_sizes'][split]
            
            if not bbox_sizes:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[i].set_title(f'{split.capitalize()} Bbox Sizes')
                continue
            
            widths = [s[0] for s in bbox_sizes]
            heights = [s[1] for s in bbox_sizes]
            
            axes[i].scatter(widths, heights, alpha=0.5, s=10)
            axes[i].set_xlabel('Width (normalized)')
            axes[i].set_ylabel('Height (normalized)')
            axes[i].set_title(f'{split.capitalize()} Bbox Sizes (n={len(bbox_sizes)})')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bbox_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_image_stats(self):
        """Create plots for dataset summary statistics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        splits = ['train', 'val', 'test']
        
        # Dataset summary bars
        splits_data = []
        images_data = []
        objects_data = []
        
        for split in splits:
            splits_data.append(split)
            images_data.append(self.stats['total_images'][split])
            objects_data.append(self.stats['total_objects'][split])
        
        x = np.arange(len(splits_data))
        width = 0.35
        
        axes[0].bar(x - width/2, images_data, width, label='Images', alpha=0.8)
        axes[0].bar(x + width/2, objects_data, width, label='Objects', alpha=0.8)
        axes[0].set_xlabel('Split')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Dataset Summary')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(splits_data)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Objects per image box plot
        data_for_boxplot = []
        labels_for_boxplot = []
        for split in splits:
            if self.stats['objects_per_image'][split]:
                data_for_boxplot.append(self.stats['objects_per_image'][split])
                labels_for_boxplot.append(split)
        
        if data_for_boxplot:
            axes[1].boxplot(data_for_boxplot, labels=labels_for_boxplot)
            axes[1].set_ylabel('Objects per Image')
            axes[1].set_title('Objects per Image Distribution')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        report = f"Dataset Statistics Report: {self.dataset_name}\n"
        report += "=" * 50 + "\n\n"
        
        # Dataset overview
        report += "Dataset Overview:\n"
        report += f"Number of classes: {self.config['nc']}\n"
        report += f"Classes: {', '.join(self.config.get('names', []))}\n\n"
        
        # Split statistics
        for split in ['train', 'val', 'test']:
            if self.stats['total_images'][split] > 0:
                report += f"{split.capitalize()} Split:\n"
                report += f"  Images: {self.stats['total_images'][split]}\n"
                report += f"  Objects: {self.stats['total_objects'][split]}\n"
                
                if self.stats['objects_per_image'][split]:
                    avg_objects = np.mean(self.stats['objects_per_image'][split])
                    report += f"  Avg objects per image: {avg_objects:.2f}\n"
                
                # Class distribution for this split
                class_counts = self.stats['class_counts'][split]
                if class_counts:
                    report += f"  Class distribution:\n"
                    for class_id, count in sorted(class_counts.items()):
                        class_name = self.config.get('names', [])[class_id] if class_id < len(self.config.get('names', [])) else f"Class {class_id}"
                        report += f"    {class_name}: {count}\n"
                
                report += "\n"
        
        # Save report
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report)
        
        # Save detailed statistics as JSON
        stats_json = {}
        for key, value in self.stats.items():
            if isinstance(value, dict):
                stats_json[key] = {}
                for split, split_value in value.items():
                    if isinstance(split_value, (list, tuple)):
                        stats_json[key][split] = len(split_value)  # Store count for lists
                    elif isinstance(split_value, Counter):
                        stats_json[key][split] = dict(split_value)
                    else:
                        stats_json[key][split] = split_value
        
        with open(self.output_dir / 'detailed_stats.json', 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(report)
    
    def run(self):
        """Run the complete analysis"""
        print(f"Starting analysis of dataset: {self.dataset_name}")
        
        # Analyze dataset
        self.analyze_dataset()
        
        # Generate visualizations
        print("Generating visualizations...")
        self.plot_bbox_center_distribution()
        self.plot_class_distribution()
        self.plot_objects_per_image()
        self.plot_bbox_size_distribution()
        self.plot_image_stats()
        
        # Generate reports
        print("Generating reports...")
        self.generate_summary_report()
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze object detection dataset statistics')
    parser.add_argument('--data_path', help='Path to the dataset YAML configuration file')
    args = parser.parse_args()

    assert os.path.exists(args.data_path), f"Error: YAML file {args.data_path} not found"

    # Create and run analyzer
    analyzer = DatasetStatsChecker(args.data_path)
    analyzer.run()

if __name__ == "__main__":
    main()
