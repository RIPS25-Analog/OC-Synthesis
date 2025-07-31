from ultralytics import YOLO
import argparse
from ultralytics import settings

class YOLOfinetuner:
    def __init__(self, **kwargs):
        self.model = YOLO(kwargs.get('model', 'yolo11n.pt'), task='detect')
        print(self.model.info())

        self.data = kwargs.get('data')
        if not kwargs.get('project', None):
            kwargs['project'] = '/home/wandb-runs/' + self.data.split('/')[-1].split('.')[0]

        if 'model' in kwargs:
            del kwargs['model']
            
        self.train_params = kwargs
        
    def train_model(self):
        results = self.model.train(**self.train_params)
        return results     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a YOLO model with specified parameters.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to the YOLO model file.')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset configuration file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--project', type=str, default=None, help='Project name for saving results.')
    
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
    parser.add_argument('--batch', type=int, default=16, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size.')
    parser.add_argument('--multi_scale', action='store_true', help='Enable multi-scale training.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging.')
    parser.add_argument('--val', type=bool, default=True, help='Enable validation during training.')
    parser.add_argument('--close_mosaic', type=int, default=10, help='Close mosaic augmentation N epochs before training ends.')
    parser.add_argument('--name', type=str, default=None, help='Name of the training run.')

    args = parser.parse_args()

    if args.no_wandb:
        print("Intra-run Weights & Biases logging is disabled.")
        settings.update({"wandb": False}) # enable WandB for standalone finetuning (not sweeps)
    else:
        settings.update({"wandb": True})
    del args.no_wandb  # Remove no_wandb from args to avoid passing it to YOLO
    

    args.val = not args.dont_val  # Convert dont_val to val
    del args.dont_val  # Remove dont_val from args

    # Convert args into dictionary
    train_kwargs = vars(args)
    finetuner = YOLOfinetuner(**train_kwargs)
    results = finetuner.train_model()
    # print(results)  # Print the training results