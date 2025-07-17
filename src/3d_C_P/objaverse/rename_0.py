import os
import glob

root = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/objaverse/obj"

# Match only files that end with _0.mtl
pattern = os.path.join(root, '**', '*_0.mtl')

for filepath in glob.glob(pattern, recursive=True):
    dir_name = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # Remove "_0" before the extension
    new_filename = filename.replace('_0.mtl', '.mtl')
    new_path = os.path.join(dir_name, new_filename)
    
    print(f"Renaming: {filepath} → {new_path}")
    os.rename(filepath, new_path)
