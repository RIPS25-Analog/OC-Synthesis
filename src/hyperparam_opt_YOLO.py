import argparse
import wandb
import os
from ultralytics import settings
from finetune_YOLO import YOLOFinetuner
from evaluate_YOLO import YOLOEvaluator

runs_save_dir = '/home/wandb-runs'
def train_with_wandb(config=None):
    """Training function to be called by WandB sweep agent."""

    with wandb.init(config=config) as run:
        config = wandb.config

        local_project_name = f"{runs_save_dir}/{project_name}/{sweep_name}/{sweep_id}"
        print(f"Running hyperparameter optimization in project {local_project_name} with config: {config}")
        finetuner = YOLOFinetuner(**config, name=run.name, project=local_project_name)
        results_train = finetuner.train_model()
        
        evaluator_args = {
            'run': str(results_train.save_dir),
            'batch': config.get('batch', 32),
            'imgsz': config.get('eval_imgsz', 640),
            'project': local_project_name,
            'split': 'val'  # Assuming we want to evaluate on the validation set
        }
        evaluator = YOLOEvaluator(**evaluator_args)
        val_results = evaluator.evaluate_model()

        metrics_to_log = {}
        
        # Extract common YOLO metrics
        for key, value in val_results.get('metrics').items():
            clean_key = key.replace('metrics/', '').replace('(B)', '')
            metrics_to_log[clean_key] = value
        
        # Reinitialize WandB since Ultralytics internally initialized and finished a run
        wandb.init(reinit=True, config=config)
        wandb.log(metrics_to_log)

def run_hyperparameter_optimization(project, data, model, sweep_count=50, epochs=20, sweep_name='yolo_hyperparam_opt', data_fraction=1.0, workers=16):
    global sweep_id
    """Run hyperparameter optimization using WandB sweeps."""
    
    # Create sweep configuration
    sweep_config = {
        'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
        'metric': {'name': 'mAP50', 'goal': 'maximize'},
        'parameters': {
            'optimizer': {'values': ['Adam']},
            # Learning rate parameters
            'lr0': {'distribution': 'log_uniform_values', 'min': 3e-5, 'max': 3e-3},
            # 'lrf': {'distribution': 'uniform', 'min': 0.001, 'max': 0.1},
            # 'momentum': {'distribution': 'uniform', 'min': 0.85, 'max': 0.95},
            # 'weight_decay': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.001},

            # # Warmup parameters
            # 'warmup_epochs': {'distribution': 'uniform', 'min': 1.0, 'max': 5.0},
            # 'warmup_momentum': {'distribution': 'uniform', 'min': 0.5, 'max': 0.9},
            # 'warmup_bias_lr': {'distribution': 'uniform', 'min': 0.05, 'max': 0.2},

            # # Loss function weights
            # 'box': {'distribution': 'uniform', 'min': 1.0, 'max': 20.0},
            # 'cls': {'distribution': 'uniform', 'min': 0.05, 'max': 2.0},
            # 'dfl': {'distribution': 'uniform', 'min': 0.1, 'max': 5.0},
            
            'close_mosaic': {'values': [5]},  # Close mosaic augmentation N epochs before training ends
            # Training parameters
            'batch': {'values': [32]},
            'imgsz': {'values': [480, 640, 800, 960]},
            'eval_imgsz': {'values': [480, 640, 800, 960]},
            'multi_scale': {'values': [0, 1]}, # making numeric for ease of plotting in WandB
            'epochs': {'values': [5, 10, 20]},
            
            # Architecture parameters
            'freeze': {'values': ([16, 19, 22] if 'world' in model else [17, 20, 23])}#{'distribution': 'int_uniform', 'min': 10, 'max': 20},
        }
    }
    
    # Add fixed parameters that don't change across runs
    if sweep_name is not None:
        sweep_config['name'] = sweep_name
        
    sweep_config['parameters']['data'] = {'value': data}
    sweep_config['parameters']['model'] = {'value': model}
    sweep_config['parameters']['val'] = {'value': False}
    sweep_config['parameters']['workers'] = {'value': workers}
    sweep_config['parameters']['fraction'] = {'value': data_fraction}
    # sweep_config['parameters']['epochs'] = {'value': epochs}  # Fixed number of epochs for all runs

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project)
    
    print(f"Starting hyperparameter optimization sweep: {sweep_id} under WandB project {project}")
    print(f"Number of runs: {sweep_count}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=sweep_count)
    
    print("Hyperparameter optimization completed!")

if __name__ == "__main__":
    global project_name, sweep_name
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for YOLO model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to the YOLO model file.')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset configuration file.')
    parser.add_argument('--fraction', type=float, default=100, help='Fraction of the dataset to use for training.')
    parser.add_argument('--workers', type=int, default=16, help='Number of workers for data loading.')
    
    # parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training in each hyperparameter run.')
    parser.add_argument('--sweep_count', type=int, default=50, help='Number of hyperparameter combinations to try.')
    parser.add_argument('--project', type=str, default='{dataset_name}', help='WandB project name for hyperparameter optimization.')
    parser.add_argument('--sweep_name', type=str, default=None, help='Name of the WandB sweep.')
    args = parser.parse_args()
    
    # Validate required arguments
    assert os.path.exists(args.data), f"Data configuration file not found: {args.data}"

    if args.project == '{dataset_name}':
        args.project = f'{args.data.split("/")[-1].split(".")[0]}'
    
    args.fraction /= 100 # Convert percentage to fraction for YOLO
    project_name = args.project
    sweep_name = args.sweep_name
    settings.update({"wandb": True}) ## to make sure intra-sweep (epoch-wise) logging is enabled

    # Use default hyperparameter optimization
    run_hyperparameter_optimization(
        project=args.project,
        data=args.data,
        model=args.model,
        sweep_count=args.sweep_count,
        # epochs=args.epochs,
        sweep_name=args.sweep_name,
        data_fraction=args.fraction,
        workers=args.workers
    )