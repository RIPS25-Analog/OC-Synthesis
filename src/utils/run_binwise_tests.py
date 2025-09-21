import os
from pathlib import Path
from collections import defaultdict
import yaml
from tqdm import tqdm
import subprocess
import pandas as pd

def run_bin_test(model_path, model_description, bin_type, bin_names):
    results = []
    assert bin_type in ['occlusion', 'brightness', 'objects'], "Invalid bin type"
    print(f"Evaluating {bin_type} bins...")
    
    for bin_name in bin_names:
        # Create safe bin name for filesystem
        if bin_type == 'objects':
            safe_bin_name = bin_name
            dataset_dir = bin_datasets_dir / f"objects_{safe_bin_name}"
            prefix = 'obj'
        else:
            safe_bin_name = bin_name.replace('%', 'pct').replace('.', '_').replace('-', '_to_')
            dataset_dir = bin_datasets_dir / f"{bin_type}_{safe_bin_name}"
            prefix = 'occ' if bin_type == 'occlusion' else 'bri'
        
        yaml_file = dataset_dir / "dataset.yaml"
        results_yaml_path = Path(f"/home/wandb-runs/bin_results/{model_description}/{prefix}_{safe_bin_name}/val/simple_evaluation_results.yaml")

        if results_yaml_path.exists():
            print(f"Results already exist for bin {safe_bin_name}, skipping evaluation.")
        else:	
            # Run evaluation
            cmd = f"python src/evaluate_YOLO.py --model {model_path} --data {yaml_file} --project bin_results/{model_description}/{prefix}_{safe_bin_name}/ --split test"
            print(f'Executing: {cmd}')
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error running evaluation for bin {safe_bin_name}: {result.stderr}")
        
        # Read mAP50 from output YAML in /home/wandb-runs/
        assert results_yaml_path.exists(), f"Results YAML {results_yaml_path} not found for bin {safe_bin_name}"

        with open(results_yaml_path, 'r') as f:
            results_yaml = yaml.safe_load(f)
            map50 = results_yaml.get('metrics').get('metrics/mAP50(B)')
            results.append(map50)
            print(f"{bin_type.capitalize()} {bin_name}: mAP50 = {map50}")
    
    return results

if __name__ == "__main__":
    base_dir = '/home/wandb-runs/pace-v3-h2h-yolo-vid-shuf'
    desired_models_suffixes = '-seq-1000'
    model_paths = [os.path.join(base_dir, x, 'weights/best.pt') for x in os.listdir(base_dir)]
    
    occ_bin_names = ['0.00-0.02%', '0.02-3.47%', '3.47-6.87%', '6.87-8.34%', '8.34-10.70%', '10.70-26.12%']
    obj_bin_names = ['1', '2', '3']
    brightness_bin_names = ['108.8-113.6', '113.6-116.7', '116.7-120.0', '120.0-171.4', '77.5-108.8']
    bin_datasets_dir = Path("/home/data/bin_datasets/")

    results = defaultdict(dict)
    for model_path in tqdm(model_paths):
        model_description = model_path.split('/')[4]
        if not model_description.endswith(desired_models_suffixes):
            continue

        results[model_path]["occlusion"] = run_bin_test(model_path, model_description, "occlusion", occ_bin_names)
        results[model_path]["object_count"] = run_bin_test(model_path, model_description, "objects", obj_bin_names)
        results[model_path]["brightness"] = run_bin_test(model_path, model_description, "brightness", brightness_bin_names)

    # Save results to CSV
    occ_df = pd.DataFrame({model_path: results[model_path]["occlusion"] for model_path in model_paths}, index=occ_bin_names)
    obj_df = pd.DataFrame({model_path: results[model_path]["object_count"] for model_path in model_paths}, index=obj_bin_names)
    brightness_df = pd.DataFrame({model_path: results[model_path]["brightness"] for model_path in model_paths}, index=brightness_bin_names)
    occ_df.to_csv("occlusion_bin_results.csv")
    obj_df.to_csv("object_count_bin_results.csv")
    brightness_df.to_csv("brightness_bin_results.csv")