import gc
import torch
import wandb
from ultralytics import YOLO

class YOLOfinetuner:
    def __init__(self, **kwargs):
        self.model = YOLO(kwargs.get('model_path', 'yolo11n.pt'), task='detect')
        print(self.model.info())
        self.data_path = kwargs.get('data_path')

        del kwargs['model_path']
        del kwargs['data_path']
        self.train_params = kwargs

    def train_model(self):
        project_name = '/home/wandb-runs/' + self.data_path.split('/')[-1].split('.')[0]
        
        # Prepare training parameters
        train_args = {
            'data': self.data_path,
            'project': project_name
        }
        
        # Add additional hyperparameters
        train_args.update(self.train_params)
        
        results = self.model.train(**train_args)
        return results

def train_with_wandb(config=None):
    """Training function to be called by WandB sweep agent."""

    with wandb.init(config=config):
        config = wandb.config
    
        train_kwargs = {
            'batch': config.batch,
            'imgsz': config.imgsz,
            'epochs': config.epochs,
            'freeze': config.freeze,
            'nbs': 64,
            'val': False, # disable periodic validation during training
        }
        
        finetuner = YOLOfinetuner(
            model_path=config.model_path,
            data_path=config.data_path,
            **train_kwargs
        )
        
        val_results = finetuner.train_model() # model evaluated on some arbitrary validation set
        
        metrics_to_log = {}
        
        # Extract common YOLO metrics
        for key, value in val_results.results_dict.items():
            clean_key = key.replace('metrics/', '').replace('(B)', '')
            metrics_to_log[clean_key] = value
        
        wandb.log(metrics_to_log)
    
    # del finetuner, val_results
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()

def run_hyperparameter_optimization(project_name, data_path, model_path, sweep_count=50, epochs=20, batch_sizes=[8], sweep_name='Sweep'):
    """Run hyperparameter optimization using WandB sweeps."""
    
    # Create sweep configuration
    sweep_config = {
        'name': sweep_name,
        'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
        'metric': {'name': 'mAP50', 'goal': 'maximize'},
        'parameters': {
            'batch': {'values': batch_sizes},
            'imgsz': {'values': [640]},
            'freeze': {'values': [23]}
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
    project_name = 'CutNPaste-memleak-investigation'
    data_path = 'cnp-v0-15365.yaml'
    model_path = 'yolo11n.pt'
    sweep_count = 5
    epochs = 1
    batch_sizes = [8,16,32,64,128]

    # ## Run in increasing batch size order
    # run_hyperparameter_optimization(
    #     project_name=project_name,
    #     data_path=data_path,
    #     model_path=model_path,
    #     sweep_count=sweep_count,
    #     epochs=epochs,
    #     batch_sizes=batch_sizes,
    #     sweep_name='increasing-batch-sizes'
    # )

    # ## Run in decreasing batch size order
    # run_hyperparameter_optimization(
    #     project_name=project_name,
    #     data_path=data_path,
    #     model_path=model_path,
    #     sweep_count=sweep_count,
    #     epochs=epochs,
    #     batch_sizes=batch_sizes[::-1],
    #     sweep_name='decreasing-batch-sizes'
    # )
    