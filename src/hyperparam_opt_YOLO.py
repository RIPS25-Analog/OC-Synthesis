import argparse
import wandb
import yaml
import os
from finetune_YOLO import YOLOfinetuner
from evaluate_YOLO import YOLOevaluator

def train_with_wandb(config=None):
    """Training function to be called by WandB sweep agent."""

    with wandb.init(config=config):
        config = wandb.config
    
        train_kwargs = {
            # 'lr0': config.lr0,
            # 'lrf': config.lrf,
            # 'momentum': config.momentum,
            # 'weight_decay': config.weight_decay,
            # 'warmup_epochs': config.warmup_epochs,
            # 'warmup_momentum': config.warmup_momentum,
            # 'warmup_bias_lr': config.warmup_bias_lr,
            # 'box': config.box,
            # 'cls': config.cls,
            # 'dfl': config.dfl,
            # 'nbs': config.nbs,
            # 'dropout': config.dropout,
            'batch': config.batch,
            'imgsz': config.imgsz,
            'epochs': config.epochs,
            'freeze': config.freeze,
            'multi_scale': config.multi_scale,  # Enable multi-scale training
            'val': False, # disable periodic validation during training
        }
        
        finetuner = YOLOfinetuner(
            model_path=config.model_path,
            data_path=config.data_path,
            **train_kwargs
        )
        
        train_val_results = finetuner.train_model() # model evaluated on some arbitrary validation set
        
        evaluator = YOLOevaluator(model=finetuner.model,
                                  data_path=config.data_path,
                                  data_split='val',
                                  eval_img_size=config.eval_img_size)  # Evaluate on validation set
        
        val_results = evaluator.evaluate_model()  # Evaluate the model
        
        del finetuner # maybe clear memory?

        metrics_to_log = {}
        
        # Extract common YOLO metrics
        for key, value in val_results.results_dict.items():
            clean_key = key.replace('metrics/', '').replace('(B)', '')
            metrics_to_log[clean_key] = value
        
        wandb.log(metrics_to_log)

def run_hyperparameter_optimization(project_name, data_path, model_path, sweep_count=50, epochs=20):
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
            
            # Training parameters
            'batch': {'values': [32]},
            'imgsz': {'values': [640, 800, 960, 1120]},
            'eval_img_size': {'values': [800, 960, 1280, 1600]},
            'multi_scale': {'values': [False, True]},

            # 'nbs': {'values': [32, 64, 128]},
            
            # Architecture parameters
            'freeze': {'values': [8, 12, 16]}#{'distribution': 'int_uniform', 'min': 10, 'max': 20},
        }
    }
    
    # Add fixed parameters that don't change across runs
    sweep_config['parameters']['data_path'] = {'value': data_path}
    sweep_config['parameters']['model_path'] = {'value': model_path}
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
    parser.add_argument('--epochs', type=int, default=20,
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