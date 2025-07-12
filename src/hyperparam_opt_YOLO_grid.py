import argparse
import wandb
import yaml
import os
from finetune_YOLO import YOLOfinetuner
from evaluate_YOLO import YOLOevaluator
import torch
import gc

def train_with_wandb(config=None):
    """Training function to be called by WandB sweep agent."""

    # try:
    with wandb.init(config=config):
        config = wandb.config
        train_kwargs = {
            'batch': config.batch,
            'imgsz': config.imgsz,
            'epochs': config.epochs,
            'val': False,
        }
        
        finetuner = YOLOfinetuner(
            model_path=config.model_path,
            data_path=config.data_path,
            **train_kwargs
        )
        
        trainval_results = finetuner.train_model() # model evaluated on some arbitrary validation set

        # evaluator = YOLOevaluator(model=finetuner.model,
        #                             data_path=config.data_path,
        #                             data_split='val')  # Evaluate on validation set
        
        val_results = trainval_results
        print(f"train-val results: {val_results.results_dict}")  # Debugging output
        # val_results = evaluator.evaluate_model()  # Evaluate the model
        
        del finetuner #, evaluator  # Clean up to free memory

        metrics_to_log = {}
        
        # Extract common YOLO metrics
        for key, value in val_results.results_dict.items():
            clean_key = key.replace('metrics/', '').replace('(B)', '')
            metrics_to_log[clean_key] = value
        
        wandb.log(metrics_to_log)
    # finally:
    #     # Comprehensive cleanup
    #     try:
    #         print('Cleaning now...')
    #         # Clear model references
    #         if 'finetuner' in locals():
    #             del finetuner.model
    #             del finetuner
    #         if 'evaluator' in locals():
    #             del evaluator.model
    #             del evaluator
    #         if 'trainval_results' in locals():
    #             del trainval_results
    #         if 'val_results' in locals():
    #             del val_results
            
    #         # Force garbage collection
    #         gc.collect()
            
    #         # Clear PyTorch cache if using GPU
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #             torch.cuda.synchronize()
            
    #     except Exception as cleanup_error:
    #         print(f"Error during cleanup: {cleanup_error}")

def run_hyperparameter_optimization(project_name, data_path, model_path, sweep_count=50, epochs=100):
    """Run hyperparameter optimization using WandB sweeps."""
    
    # Create sweep configuration
    sweep_config = {
        'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
        'metric': {'name': 'mAP50', 'goal': 'maximize'},
        'parameters': {
            # Learning rate parameters
            'batch': {'values': list(range(32,38))},
            'imgsz': {'values': [928]},
            # 'batch': {'values': [48, 32, 16]},
            # 'imgsz': {'values': [420, 640, 920]},
            'nbs': {'values': [32]},
        }
    }
    
    # Add fixed parameters that don't change across runs
    sweep_config['parameters']['data_path'] = {'value': data_path}
    sweep_config['parameters']['model_path'] = {'value': model_path}
    sweep_config['parameters']['freeze'] = {'value': 23}  # Fixed freeze value
    sweep_config['parameters']['epochs'] = {'value': epochs}  # Fixed number of epochs for all runs

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    print(f"Starting hyperparameter optimization sweep: {sweep_id} under WandB project {project_name}")
    print(f"Number of runs: {sweep_count}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=sweep_count)
    
    print("Hyperparameter optimization completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for YOLO model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--project_name', type=str, default='runs-{dataset_name}', 
                       help='WandB project name for hyperparameter optimization.')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to the dataset configuration file.')
    parser.add_argument('--model_path', type=str, default='yolo11n.pt', 
                       help='Path to the YOLO model file.')
    parser.add_argument('--sweep_count', type=int, default=50, 
                       help='Number of hyperparameter combinations to try.')
    parser.add_argument('--metric_name', type=str, default='final_mAP50',
                       help='Metric to optimize (final_mAP50, final_mAP50-95).')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for training in each hyperparameter run.')
    
    args = parser.parse_args()
    
    # Validate required arguments
    assert os.path.exists(args.data_path), f"Data configuration file not found: {args.data_path}"

    if args.project_name == 'runs-{dataset_name}':
        args.project_name = f'runs-{args.data_path.split("/")[-1].split(".")[0]}'

    # Use default hyperparameter optimization
    run_hyperparameter_optimization(
        project_name=args.project_name,
        data_path=args.data_path,
        model_path=args.model_path,
        sweep_count=args.sweep_count,
        epochs=args.epochs
    )