from ultralytics import YOLO
import argparse
from ultralytics import settings

settings.update({"wandb": False}) # disable WandB so it doesn't interfere if doing hyperparameter sweeps

class YOLOfinetuner:
    def __init__(self, **kwargs):
        self.model = YOLO(kwargs.get('model_path', 'yolo11n.pt'), task='detect')
        print(self.model.info())
        self.data_path = kwargs.get('data_path')
        self.epochs = kwargs.get('epochs')
        self.freeze = kwargs.get('freeze')

        del kwargs['model_path']
        del kwargs['data_path']
        self.train_params = kwargs

    def train_model(self):
        project_name = 'runs/' + self.data_path.split('/')[-1].split('.')[0]
        
        # Prepare training parameters
        train_args = {
            'data': self.data_path,
            'project': project_name
        }
        
        # Add additional hyperparameters
        train_args.update(self.train_params)
        
        results = self.model.train(**train_args)
        return results
    
    
    
if __name__ == "__main__":
    settings.update({"wandb": True}) # enable WandB for standalone finetuning (not sweeps)
    
    parser = argparse.ArgumentParser(description='Train a YOLO model with specified parameters.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='yolo11n.pt', help='Path to the YOLO model file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset configuration file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    
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
    
    args = parser.parse_args()

    # Convert args into dictionary
    train_kwargs = vars(args)
    finetuner = YOLOfinetuner(**train_kwargs)
    results = finetuner.train_model()
    print(results)  # Print the training results