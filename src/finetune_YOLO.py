import os
import wandb
import yaml
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.files import WorkingDirectory
import argparse
from ultralytics import settings

wandb_prefix = 'vikhyat-3-org/pace-v3/'

class YOLOFinetuner:
    def __init__(self, **kwargs):
        model_name = kwargs.get('model', 'yolo11n.pt')
        if 'world' in model_name:
            self.model = YOLOWorld(model_name)

            with open(kwargs['data'], 'r') as f:
                data = yaml.safe_load(f.read())
            data_names = data.get('names', None)
            if isinstance(data_names, list):
                self.class_names = [x.replace('_',' ') for x in data_names]
            elif isinstance(data_names, dict):
                self.class_names = [data_names[i].replace('_',' ') for i in sorted(data_names.keys())]
            else:
                raise ValueError("Invalid 'names' format in data config.")
            print('Using class names:', self.class_names)
            self.model.set_classes(self.class_names)
        else:
            self.model = YOLO(model_name, task='detect')
        print(self.model.info())

        self.data = kwargs.get('data')

        if 'model' in kwargs:
            del kwargs['model']
        if 'eval_imgsz' in kwargs:
            del kwargs['eval_imgsz']
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = 'Adam'  # Use Adam optimizer by default
        if 'multi_scale' in kwargs:
            kwargs['multi_scale'] = bool(kwargs['multi_scale'])

        if 'save_dir' in kwargs:
            self.save_dir = kwargs['save_dir']
            del kwargs['save_dir']  # Remove save_dir from kwargs to avoid passing it to YOLO
        else:
            self.save_dir = os.path.join('/home/wandb-runs') # project_name/run_name gets added automatically
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_params = kwargs
        
    def train_model(self):
        with WorkingDirectory(self.save_dir):
            results = self.model.train(**self.train_params)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a YOLO model with specified parameters.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to the YOLO model file.')
    parser.add_argument('--parent_sweep_name_dir', type=str, default=None, help='Start model training from the best model found in a given sweep directory.')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset configuration file.')
    parser.add_argument('--project', type=str, default=None, help='Project name for saving results.')
    parser.add_argument('--name', type=str, default=None, help='Name of the training run.')   

    parser.add_argument('--workers', type=int, default=16, help='Number of workers for data loading.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging.')
    parser.add_argument('--val', type=bool, default=True, help='Enable validation during training.')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of the dataset to use for training.')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=32, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size.')
    parser.add_argument('--multi_scale', action='store_true', help='Enable multi-scale training.')
    parser.add_argument('--close_mosaic', type=int, default=10, help='Close mosaic augmentation N epochs before training ends.')
    
    # Additional hyperparameters
    parser.add_argument('--freeze', type=int, default=23, help='Number of layers to freeze during training.')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor.')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimizer weight decay.')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='Warmup epochs.')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='Warmup initial momentum.')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='Warmup initial bias learning rate.')
    parser.add_argument('--box', type=float, default=7.5, help='Box loss gain.')
    parser.add_argument('--cls', type=float, default=0.5, help='Classification loss gain.')
    parser.add_argument('--dfl', type=float, default=1.5, help='Distribution focal loss gain.')
    parser.add_argument('--nbs', type=int, default=64, help='Nominal batch size.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Use dropout regularization.')
    
    args = parser.parse_args()

    if args.no_wandb:
        print("Intra-run Weights & Biases logging is disabled.")
        settings.update({"wandb": False}) # enable WandB for standalone finetuning (not sweeps)
    else:
        settings.update({"wandb": True})
    del args.no_wandb  # Remove no_wandb from args to avoid passing it to YOLO

    if args.project is None:
        args.project = args.data.split('/')[-1].split('.')[0]

    if args.parent_sweep_name_dir is not None:
        assert args.model==parser.get_default('model'), "Cannot specify model when parent_sweep_name_dir being used"
        # Find best weights for best sweep in given sweep name directory
        sweep_name_dir = args.parent_sweep_name_dir
        sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
        assert len(sweep_ids)==1, f"{len(sweep_ids)} sweeps found in {sweep_name_dir}, unsure which to use, so skipping"
        sweep_id = sweep_ids[0]
        
        wandb_api = wandb.Api()
        sweep = wandb_api.sweep(wandb_prefix + sweep_id)
        best_run = sorted(sweep.runs, key=lambda run: run.summary.get("mAP50", 0), reverse=True)[0]
        args.model = os.path.join(sweep_name_dir, sweep_id, best_run.name + '_extended', 'weights', 'best.pt')
        
    del args.parent_sweep_name_dir

    args.fraction /= 100 # Convert percentage to fraction for YOLO
    
    # Convert args into dictionary
    train_kwargs = vars(args)
    finetuner = YOLOFinetuner(**train_kwargs)
    results = finetuner.train_model()
    # print(results)  # Print the training results