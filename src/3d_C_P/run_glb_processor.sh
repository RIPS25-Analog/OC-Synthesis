cd /home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/

# Save the trimesh processor
cat > trimesh_glb_processor.py << 'EOF'
#!/usr/bin/env python3
"""
GLB processor using trimesh - processes ALL files, only creates labeled variants
"""

import trimesh
import os
import glob
import json
import numpy as np
from pathlib import Path

class TrimeshGLBProcessor:
    def __init__(self, input_base_dir, output_base_dir):
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        
        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Standard sizes for each tool category (in meters)
        self.standard_sizes = {
            'hammer': 0.30,        # 30cm typical hammer
            'screwdriver': 0.20,   # 20cm typical screwdriver
            'pliers': 0.18,        # 18cm typical pliers
            'wrench': 0.25,        # 25cm typical wrench
            'tape_measure': 0.10,  # 10cm typical tape measure
            'utility_knife': 0.15, # 15cm typical utility knife
            'drill': 0.25,         # 25cm typical drill
        }
        
        # Size variants - REMOVED unlabeled base, only keep labeled variants
        self.size_variants = {
            'small': 0.6,      # 70% of standard
            'medium': 1.0,     # 100% of standard
            'large': 1.8       # 130% of standard
        }
    
    def normalize_mesh_size(self, mesh, tool_category):
        """Normalize mesh to standard size for its category"""
        if mesh is None or mesh.is_empty:
            return None
        
        target_size = self.standard_sizes.get(tool_category, 0.20)
        current_max_dim = max(mesh.extents)
        
        if current_max_dim > 0:
            scale_factor = target_size / current_max_dim
            mesh.apply_scale(scale_factor)
            print(f"  📏 Normalized from {current_max_dim:.3f}m to {target_size:.3f}m (scale: {scale_factor:.3f})")
        
        return mesh
    
    def process_single_glb_file(self, glb_path, tool_category):
        """Process a single GLB file using trimesh"""
        print(f"\n🔧 Processing: {os.path.basename(glb_path)}")
        
        try:
            # Load GLB with trimesh
            mesh = trimesh.load(glb_path)
            
            if mesh is None:
                print(f"❌ Failed to load mesh from {glb_path}")
                return []
            
            # Handle scene vs single mesh
            if isinstance(mesh, trimesh.Scene):
                print(f"📦 Loaded scene with {len(mesh.geometry)} geometries")
                # Combine all geometries into one mesh
                geometries = []
                for name, geom in mesh.geometry.items():
                    if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                        geometries.append(geom)
                
                if geometries:
                    mesh = trimesh.util.concatenate(geometries)
                else:
                    print(f"❌ No valid geometries found in scene")
                    return []
            
            print(f"📐 Original dimensions: {mesh.extents}")
            
            # Normalize size to standard for this tool category
            normalized_mesh = self.normalize_mesh_size(mesh.copy(), tool_category)
            if normalized_mesh is None:
                print(f"❌ Failed to normalize mesh")
                return []
            
            # Get original filename without extension
            original_name = Path(glb_path).stem
            
            # Create output directory
            output_category_dir = os.path.join(self.output_base_dir, tool_category, "models")
            os.makedirs(output_category_dir, exist_ok=True)
            
            # REMOVED: No longer save unlabeled normalized base model
            # base_output_path = os.path.join(output_category_dir, f"{original_name}.glb")
            
            # Create ONLY labeled size variants
            variants_created = []
            variant_count = 0
            
            print(f"  🎨 Creating labeled size variants...")
            
            for size_name, size_multiplier in self.size_variants.items():
                # Create variant mesh from normalized base
                variant_mesh = normalized_mesh.copy()
                
                if size_name != 'medium':  # Medium is already at standard size
                    variant_mesh.apply_scale(size_multiplier)
                
                # Save variant with size label
                variant_filename = f"{original_name}_{size_name}.glb"
                variant_path = os.path.join(output_category_dir, variant_filename)
                
                variant_mesh.export(variant_path)
                
                variants_created.append({
                    'variant_id': variant_count,
                    'filename': variant_filename,
                    'size': size_name,
                    'size_multiplier': size_multiplier,
                    'source_file': os.path.basename(glb_path),
                    'dimensions': variant_mesh.extents.tolist(),
                    'max_dimension': max(variant_mesh.extents)
                })
                
                variant_count += 1
                print(f"    ✅ Created {size_name} variant: {variant_filename} (max dim: {max(variant_mesh.extents):.3f}m)")
            
            print(f"  ✅ Created {len(variants_created)} labeled variants")
            return variants_created
            
        except Exception as e:
            print(f"❌ Error processing {glb_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_all_tools(self):
        """Process ALL GLB files in the input directory"""
        print(f"🚀 Starting GLB processing with trimesh...")
        print(f"📁 Input: {self.input_base_dir}")
        print(f"📁 Output: {self.output_base_dir}")
        print(f"🎯 Processing ALL files (not just first 3)")
        print(f"📋 Creating only labeled variants (small, medium, large)")
        
        all_results = {}
        
        # Find all tool categories
        if not os.path.exists(self.input_base_dir):
            print(f"❌ Input directory not found: {self.input_base_dir}")
            return {}
        
        tool_dirs = [d for d in os.listdir(self.input_base_dir) 
                    if os.path.isdir(os.path.join(self.input_base_dir, d))]
        
        if not tool_dirs:
            print(f"❌ No tool directories found in {self.input_base_dir}")
            return {}
        
        grand_total_files = 0
        grand_total_variants = 0
        
        for tool_category in tool_dirs:
            print(f"\n🔧 Processing category: {tool_category}")
            
            # Find all GLB files in this category
            models_dir = os.path.join(self.input_base_dir, tool_category, "models")
            
            if not os.path.exists(models_dir):
                print(f"❌ Models directory not found: {models_dir}")
                continue
            
            glb_files = glob.glob(os.path.join(models_dir, "*.glb"))
            
            if not glb_files:
                print(f"❌ No GLB files found in {models_dir}")
                continue
            
            print(f"📄 Found {len(glb_files)} GLB files")
            grand_total_files += len(glb_files)
            
            category_results = []
            
            # Process ALL files (removed [:3] limit)
            for i, glb_path in enumerate(glb_files):
                print(f"\n--- Processing file {i+1}/{len(glb_files)} in {tool_category} ---")
                variants = self.process_single_glb_file(glb_path, tool_category)
                category_results.extend(variants)
                grand_total_variants += len(variants)
            
            all_results[tool_category] = category_results
            print(f"✅ {tool_category}: {len(category_results)} total variants created from {len(glb_files)} files")
        
        # Save detailed summary
        summary = {
            'input_directory': self.input_base_dir,
            'output_directory': self.output_base_dir,
            'standard_sizes': self.standard_sizes,
            'size_variants': self.size_variants,
            'processing_summary': {
                'total_input_files': grand_total_files,
                'total_output_variants': grand_total_variants,
                'variants_per_file': len(self.size_variants),
                'categories_processed': len(all_results)
            },
            'results': all_results
        }
        
        summary_path = os.path.join(self.output_base_dir, "trimesh_processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Processing complete!")
        print(f"📊 Total input files processed: {grand_total_files}")
        print(f"📊 Total output variants created: {grand_total_variants}")
        print(f"📊 Variants per input file: {len(self.size_variants)} (small, medium, large)")
        print(f"📄 Summary saved: {summary_path}")
        
        return all_results

def main():
    """Main function"""
    input_dir = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Objaverse_data"
    output_dir = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Objaverse_data_resized"
    
    processor = TrimeshGLBProcessor(input_dir, output_dir)
    results = processor.process_all_tools()
    
    # Print detailed final summary
    print(f"\n📋 DETAILED FINAL SUMMARY:")
    total_variants = 0
    for tool_category, variants in results.items():
        count = len(variants)
        files_processed = count // 3  # Since we create 3 variants per file
        total_variants += count
        print(f"  {tool_category}: {count} variants from {files_processed} files")
    
    print(f"\n🎯 GRAND TOTAL: {total_variants} variants created")
    
    # Show some example output files
    if results:
        print(f"\n📁 Example output files:")
        for category, variants in results.items():
            if variants:
                print(f"  {category}:")
                for variant in variants[:3]:  # Show first 3 variants
                    print(f"    - {variant['filename']} ({variant['size']}, {variant['max_dimension']:.3f}m)")
                if len(variants) > 3:
                    print(f"    ... and {len(variants) - 3} more")
                break

if __name__ == "__main__":
    main()
EOF

# Run with your virtual environment (which has trimesh)
source /home/coraguo/RIPS25-AnalogDevices-ObjectDetection/env/bin/activate
python trimesh_glb_processor.py