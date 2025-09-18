import argparse
import wandb
import os
from ultralytics import settings
from finetune_YOLO import YOLOFinetuner
from evaluate_YOLO import YOLOEvaluator

runs_save_dir = '/home/wandb-runs'
wandb_prefix = 'vikhyat-3-org/'

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
            'imgsz': config.get('imgsz', 640),
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

def run_hyperparameter_optimization(project, data, model, sweep_count=50, sweep_name='yolo_hyperparam_opt', data_fraction=1.0, workers=16):
    global sweep_id
    """Run hyperparameter optimization using WandB sweeps."""
    
    # Create sweep configuration
    sweep_config = {
        'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
        'metric': {'name': 'mAP50', 'goal': 'maximize'},
        'parameters': {
            'optimizer': {'values': ['Adam']},
            'close_mosaic': {'values': [5]},  # Close mosaic augmentation N epochs before training ends
            'batch': {'values': [32]},
            'multi_scale': {'values': [0]}, # making numeric for ease of plotting in WandB
            'epochs': {'values': [25]},

            'lr0': {'values': [1e-4,1e-3]},
            'imgsz': {'values': [640, 960, 1280]},
            'freeze': {'values': [17, 20]}#{'distribution': 'int_uniform', 'min': 10, 'max': 20},
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
    parser.add_argument('--parent_sweep_name_dir', type=str, default=None, help='Start model training from the best model found in a given sweep directory.')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset YAML config file.')
    parser.add_argument('--fraction', type=float, default=100, help='Fraction of the dataset to use for training.')
    parser.add_argument('--workers', type=int, default=16, help='Number of workers for data loading.')
    
    # parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training in each hyperparameter run.')
    parser.add_argument('--sweep_count', type=int, default=50, help='Number of hyperparameter combinations to try.')
    parser.add_argument('--project', type=str, default='{data_config_name}', help='WandB project name for hyperparameter optimization.')
    parser.add_argument('--sweep_name', type=str, default=None, help='Name of the WandB sweep.')
    args = parser.parse_args()
    
    # Validate required arguments
    assert os.path.exists(args.data), f"Data configuration file not found: {args.data}"

    if args.project == '{data_config_name}':
        args.project = f'{args.data.split("/")[-1].split(".")[0]}'

    if args.parent_sweep_name_dir is not None:
        # Find best weights for best sweep in given sweep name directory
        sweep_name_dir = args.parent_sweep_name_dir
        sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
        assert len(sweep_ids)==1, f"{len(sweep_ids)} sweeps found in {sweep_name_dir}, unsure which to use, so skipping"
        sweep_id = sweep_ids[0]

        wandb_api = wandb.Api()
        sweep = wandb_api.sweep(wandb_prefix + args.project + '/' + sweep_id)
        best_run = sorted(sweep.runs, key=lambda run: run.summary.get("mAP50", 0), reverse=True)[0]
        args.model = os.path.join(sweep_name_dir, sweep_id, best_run.name, 'weights', 'best.pt')
        assert os.path.exists(args.model), f"Best model not found: {args.model}"
        del args.parent_sweep_name_dir

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