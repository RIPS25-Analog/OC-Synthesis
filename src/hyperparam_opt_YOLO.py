import argparse
import wandb
import os
from finetune_YOLO import YOLOfinetuner
from evaluate_YOLO import YOLOevaluator

def train_with_wandb(config=None):
    """Training function to be called by WandB sweep agent."""

    with wandb.init(config=config) as run:
        config = wandb.config
    
        project_name = f"/home/wandb-runs/{config.data.split('/')[-1].split('.')[0]}/{sweep_id}"
        run_name = run.name
        finetune_args = {'data': config.data,
                        'model': config.model,
                        'epochs': config.epochs,
                        'batch': config.batch,
                        'imgsz': config.imgsz,
                        'freeze': config.freeze,
                        'project': project_name,
                        'name': run_name,
                        'val': False
                        }
        
        finetuner = YOLOfinetuner(**finetune_args)
        results_train = finetuner.train_model()
        
        evaluator_args = {
            'run': str(results_train.save_dir),
            'batch': config.batch,
            'imgsz': config.imgsz,
            'project': project_name,
            'split': 'val'  # Assuming we want to evaluate on the validation set
        }
        evaluator = YOLOevaluator(**evaluator_args)
        val_results = evaluator.evaluate_model()

        metrics_to_log = {}
        
        # Extract common YOLO metrics
        for key, value in val_results.get('metrics').items():
            clean_key = key.replace('metrics/', '').replace('(B)', '')
            metrics_to_log[clean_key] = value
        
        # Reinitialize WandB since Ultralytics internally initialized and finished a run
        wandb.init(reinit=True, config=config)
        wandb.log(metrics_to_log)

def run_hyperparameter_optimization(project_name, data, model, sweep_count=50, epochs=20, sweep_name='yolo_hyperparam_opt', classes=None):
    global sweep_id
    """Run hyperparameter optimization using WandB sweeps."""
    
    # Create sweep configuration
    sweep_config = {
        'method': 'random',  # Can be 'grid', 'random', or 'bayes'
        'metric': {'name': 'mAP50', 'goal': 'maximize'},
        'parameters': {
            # Learning rate parameters
            # 'lr0': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.1},
            # 'lrf': {'distribution': 'uniform', 'min': 0.001, 'max': 0.1},
            # 'momentum': {'distribution': 'uniform', 'min': 0.85, 'max': 0.95},
            # 'weight_decay': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.001},

            # # Warmup parameters
            # 'warmup_epochs': {'distribution': 'uniform', 'min': 1.0, 'max': 5.0},
            # 'warmup_momentum': {'distribution': 'uniform', 'min': 0.5, 'max': 0.9},
            # 'warmup_bias_lr': {'distribution': 'uniform', 'min': 0.05, 'max': 0.2},

            # # Loss function weights
            # 'box': {'distribution': 'uniform', 'min': 5.0, 'max': 10.0},
            # 'cls': {'distribution': 'uniform', 'min': 0.25, 'max': 1.0},
            # 'dfl': {'distribution': 'uniform', 'min': 1.0, 'max': 2.0},

            # # Regularization
            # 'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
            # 'nbs': {'values': [32, 64, 128]},
            
            # Training parameters
            'batch': {'values': [32]},
            'imgsz': {'values': [480, 640, 800, 960]},
            'eval_imgsz': {'values': [480, 640, 800, 960]},
            'multi_scale': {'values': [0,1]}, # making numeric for ease of plotting in WandB
            
            # Architecture parameters
            'freeze': {'values': [10, 15, 20]}#{'distribution': 'int_uniform', 'min': 10, 'max': 20},
        }
    }
    
    # Add fixed parameters that don't change across runs
    if sweep_name is not None:
        sweep_config['name'] = sweep_name
        
    sweep_config['parameters']['data'] = {'value': data}
    sweep_config['parameters']['model'] = {'value': model}
    sweep_config['parameters']['epochs'] = {'value': epochs}  # Fixed number of epochs for all runs
    if classes is not None:
        sweep_config['parameters']['classes'] = {'value': classes}  

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    print(f"Starting hyperparameter optimization sweep: {sweep_id} under WandB project {project_name}")
    print(f"Number of runs: {sweep_count}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=sweep_count)
    
    print("Hyperparameter optimization completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for YOLO model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset configuration file.')
    parser.add_argument('--project_name', type=str, default='runs-{dataset_name}', help='WandB project name for hyperparameter optimization.')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to the YOLO model file.')
    parser.add_argument('--sweep_count', type=int, default=50, help='Number of hyperparameter combinations to try.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training in each hyperparameter run.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--sweep_name', type=str, default=None, help='Name of the WandB sweep.')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated list of class names to evaluate (default: None, evaluates all classes).')
    args = parser.parse_args()
    args.classes = list(map(int, args.classes.split(','))) if args.classes else None
    # Validate required arguments
    assert os.path.exists(args.data), f"Data configuration file not found: {args.data}"

    if args.project_name == 'runs-{dataset_name}':
        args.project_name = f'runs-{args.data.split("/")[-1].split(".")[0]}'

    # Use default hyperparameter optimization
    run_hyperparameter_optimization(
        project_name=args.project_name,
        data=args.data,
        model=args.model,
        sweep_count=args.sweep_count,
        epochs=args.epochs,
        sweep_name=args.sweep_name,
        classes=args.classes
    )