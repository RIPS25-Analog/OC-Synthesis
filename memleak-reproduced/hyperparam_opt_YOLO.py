import argparse
import wandb
import subprocess

def train_with_wandb(config=None):
    """Training function to be called by WandB sweep agent."""

    with wandb.init(config=config):
        config = wandb.config

        project_name = f"/home/wandb-runs/{config.data.split('/')[-1].split('.')[0]}/{sweep_id}"
        
        subprocess.run(f'python finetune_YOLO.py --data {config.data} --model {config.model} --epochs {config.epochs}\
                        --batch {config.batch} --imgsz {config.imgsz} --freeze {config.freeze}\
                        --project {project_name} --dont_val --no_wandb', shell=True, check=True)

        # find most recently modified folder within project_name
        result = subprocess.run(f'find {project_name} -type d -name "train*" -printf "%T@ %p\\n" | sort -n | tail -1', 
                                shell=True, capture_output=True, text=True)
        print('result:', result)
        train_dir = result.stdout.split(' ')[-1].strip()
        print('result end:', train_dir)
        
        ## fetch results from last line of csv
        print('tr')
        with open(f'{train_dir}/results.csv', 'r') as f:
            lines = f.readlines()
            first_line = lines[0] # contains headers
            last_line = lines[-1]
            val_results = {k: float(v) for k, v in zip(first_line.split(','), last_line.split(','))}

        metrics_to_log = {}
        
        # Extract common YOLO metrics
        for key, value in val_results.items():
            clean_key = key.replace('metrics/', '').replace('(B)', '')
            metrics_to_log[clean_key] = value
        
        wandb.log(metrics_to_log)
    
    # del finetuner, val_results
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()

def run_hyperparameter_optimization(project_name, data, model, sweep_count=50, epochs=20, batch_sizes=[8], sweep_name='Sweep'):
    global sweep_id
    """Run hyperparameter optimization using WandB sweeps."""
    
    # Create sweep configuration
    sweep_config = {
        'name': sweep_name,
        'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
        'metric': {'name': 'mAP50', 'goal': 'maximize'},
        'parameters': {
            'batch': {'values': batch_sizes},
            'imgsz': {'values': [640]},
            'freeze': {'values': [23]},
            'dummy':{'values':list(range(10))},
        }
    }
    
    # Add fixed parameters that don't change across runs
    sweep_config['parameters']['data'] = {'value': data}
    sweep_config['parameters']['model'] = {'value': model}
    sweep_config['parameters']['epochs'] = {'value': epochs}  # Fixed number of epochs for all runs

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    print(f"Starting hyperparameter optimization sweep: {sweep_id} under WandB project {project_name}")
    print(f"Number of runs: {sweep_count}")
    
    # Run the sweep
    wandb.agent(sweep_id, train_with_wandb, count=sweep_count)
    print("Hyperparameter optimization completed!")

if __name__ == "__main__":
    project_name = 'CutNPaste-memleak-investigation'
    data = 'cnp-v0-15365.yaml'
    model = 'yolo11n.pt'
    sweep_count = 10
    epochs = 25
    batch_sizes = [32] #[8,16,32,64,128]

    run_hyperparameter_optimization(
        project_name=project_name,
        data=data,
        model=model,
        sweep_count=sweep_count,
        epochs=epochs,
        batch_sizes=batch_sizes,
        sweep_name=f'python-finetune-subproc'
    )