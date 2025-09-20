import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import json
from collections import defaultdict
import yaml
from tqdm.notebook import tqdm
import subprocess
from datetime import datetime

def run_occ_test(model_path, model_description):
    # Run YOLO evaluation on occlusion bins
    results = []
    print("Evaluating occlusion bins...")
    for bin_name in occ_bin_names:
        safe_bin_name = bin_name.replace('%', 'pct').replace('.', '_').replace('-', '_to_')
        dataset_dir = bin_datasets_dir / f"occlusion_{safe_bin_name}"
        
        dataset_dir = bin_datasets_dir / f"occlusion_{safe_bin_name}"
        yaml_file = dataset_dir / "dataset.yaml"
        results_yaml_path = Path(f"/home/wandb-runs/bin_results/{model_description}/occ_{safe_bin_name}/val/simple_evaluation_results.yaml")

        if results_yaml_path.exists():
            print(f"Results already exist for bin {safe_bin_name}, skipping evaluation.")
        else:	
            # Run evaluation
            cmd = f"python src/evaluate_YOLO.py --model {model_path} --data {yaml_file} --project bin_results/{model_description}/occ_{safe_bin_name}/ --split test"
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
            print(f"Occlusion {bin_name}: mAP50 = {map50}")
    
    return results

def run_obj_test(model_path, model_description):
    # Run YOLO evaluation on object count bins
    results = []
    print("Evaluating object count bins...")
    for bin_name in obj_bin_names:
        safe_bin_name = bin_name
        dataset_dir = bin_datasets_dir / f"objects_{safe_bin_name}"
        
        dataset_dir = bin_datasets_dir / f"objects_{safe_bin_name}"
        yaml_file = dataset_dir / "dataset.yaml"
        results_yaml_path = Path(f"/home/wandb-runs/bin_results/{model_description}/obj_{safe_bin_name}/val/simple_evaluation_results.yaml")

        if results_yaml_path.exists():
            print(f"Results already exist for bin {safe_bin_name}, skipping evaluation.")
        else:	
            # Run evaluation
            cmd = f"python src/evaluate_YOLO.py --model {model_path} --data {yaml_file} --project bin_results/{model_description}/obj_{safe_bin_name}/ --split test"
            print(f'Executing: {cmd}')
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Read mAP50 from output YAML in /home/wandb-runs/
        assert results_yaml_path.exists(), f"Results YAML {results_yaml_path} not found for bin {safe_bin_name}"

        with open(results_yaml_path, 'r') as f:
            results_yaml = yaml.safe_load(f)
            map50 = results_yaml.get('metrics').get('metrics/mAP50(B)')
            results.append(map50)
            print(f"Object Count {bin_name}: mAP50 = {map50}")
    
    return results

if __name__ == "__main__":
    # Model path to evaluate
    # model_paths = ["/home/wandb-runs/pace-v3-main-yolo11n/real-only-10/21kqg8io/astral-sweep-6/weights/best.pt",
    #                "/home/wandb-runs/pace-v3-main-yolo11n/2D_CNP-seq/tbc9hf2o/summer-sweep-12/weights/best.pt",
    #                "/home/wandb-runs/pace-v3-main-yolo11n/diffusion-seq/gpq9v4kw/wobbly-sweep-5/weights/best.pt",
    #                "/home/wandb-runs/pace-v3-main-yolo11n/3D_RP-seq/17uw3zlb/breezy-sweep-18/weights/best.pt",
    #                "/home/wandb-runs/pace-v3-main-yolo11n/3D_CopyPaste-seq/hkecsr7z/breezy-sweep-5/weights/best.pt"]
    base_dir = '/home/wandb-runs/pace-v3-h2h-yolo-vid-shuf'
    model_paths = [os.path.join(base_dir, x, 'weights/best.pt') for x in os.listdir(base_dir)]
    #### If previous sections did not run, define these variables here
    occ_bin_names = ['0.00-0.02%', '0.02-3.47%', '3.47-6.87%', '6.87-8.34%', '8.34-10.70%', '10.70-26.12%']
    obj_bin_names = ['1', '2', '3']
    bin_datasets_dir = Path("/home/data/bin_datasets/")

    results = defaultdict(dict)
    for model_path in model_paths:
        model_description = model_path.split('/')[4]

        results[model_path]["occlusion"] = run_occ_test(model_path, model_description)
        results[model_path]["object_count"] = run_obj_test(model_path, model_description)

    # Save results to CSV
    import pandas as pd
    occ_df = pd.DataFrame({model_path: results[model_path]["occlusion"] for model_path in model_paths}, index=occ_bin_names)
    obj_df = pd.DataFrame({model_path: results[model_path]["object_count"] for model_path in model_paths}, index=obj_bin_names)
    occ_df.to_csv("occlusion_bin_results.csv")
    obj_df.to_csv("object_count_bin_results.csv")