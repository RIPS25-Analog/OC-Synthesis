import os
import yaml
import argparse
from ultralytics import RTDETR
from ultralytics.utils.files import WorkingDirectory

wandb_runs_dir = '/home/va6hp/wandb-runs'
class RTDETREvaluator:
    def __init__(self, **kwargs):
        if kwargs.get('model') is None:
            run_dir = kwargs.get('run', None)
            assert run_dir is not None, "If no model is provided, run must be specified."
            self.model_path = os.path.join(run_dir, 'weights', 'best.pt')
            if kwargs.get('data', None) is None:
                kwargs['data'] = self._get_data_from_model_yaml(run_dir)
        else:
            self.model_path = kwargs.get('model')
            assert kwargs.get('data', None) is not None, "If a model is provided, data must also be specified."
        
        # RT-DETR doesn't have a "world" variant, so we remove that logic
        self.model = RTDETR(self.model_path)

        if ('project' not in kwargs) and ('save_dir' not in kwargs):
            self.save_dir = run_dir
        elif 'save_dir' in kwargs:
            self.save_dir = kwargs['save_dir']
            del kwargs['save_dir']  # Remove save_dir from kwargs to avoid passing it to RT-DETR
        else:
            self.save_dir = os.path.join(wandb_runs_dir, kwargs.get('project', 'rtdetr_finetune'))
        os.makedirs(self.save_dir, exist_ok=True)

        if 'run' in kwargs: del kwargs['run']
        if 'model' in kwargs: del kwargs['model']

        self.val_params = kwargs

    def _get_data_from_model_yaml(self, run_dir):
        """Fetch data path from the model's YAML configuration if not provided."""
        print(f'Fetching data path from the model YAML configuration in {run_dir}...')
        with open(os.path.join(run_dir, 'args.yaml'), 'r') as file:
            yaml_content = file.read()
            yaml_data = yaml.safe_load(yaml_content)
            data = yaml_data.get('data', None)
            
        assert data is not None, ValueError("Data path not found in the run's YAML config file.")
        return data

    def evaluate_model(self):
        """Evaluate the RT-DETR model and return results."""
        print(f"Evaluating on dataset: {self.val_params.get('data')} with split: {self.val_params.get('split')}")
        with WorkingDirectory(self.save_dir):
            results = self.model.val(**self.val_params)
        
        # Print the evaluation results
        print("Evaluation Results:")
        print(f"Class Names: {results.names}")
        print(f"mAP: {results.maps}")
        print(f"Number of Detections per Class: {results.nt_per_class}")
        print(f"Number of Detections per Image: {results.nt_per_image}")
        print(f"Results Dictionary: {results.results_dict}")
        print(f"Speed: {results.speed}")

        result_dict = {'class_names': results.names,
                        'mAPs': list(map(float, results.maps)),
                        'nt_per_class': list(map(float, results.nt_per_class)),
                        'nt_per_image': list(map(float, results.nt_per_image)),
                        'metrics': {k: float(v) for k, v in results.results_dict.items()},
                        'speed': results.speed,
                        'model': self.model_path if hasattr(self, 'model_path') else None,
                        'data': self.val_params.get('data')
                        }

        # Save the results to a file
        results_file = os.path.join(results.save_dir, 'simple_evaluation_results.yaml')
        with open(results_file, 'w') as file:
            yaml.dump(result_dict, file)
        
        return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an RT-DETR model with specified parameters.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run', type=str, default='rtdetr-l.pt', help='Path to the RT-DETR run directory (to fetch best weights from).')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation.')
    parser.add_argument('--project', type=str, default=None, help='Project name for saving evaluation results.')
    
    parser.add_argument('--data', type=str, required=False, help='Path to the dataset configuration file (YAML). (Optional, will be fetched from the run YAML if not provided).')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Specify the dataset split to evaluate on.')
    parser.add_argument('--model', type=str, default=None, help='Path to the RT-DETR model file (optional). If provided, data must also be specified.')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated list of class names to evaluate (default: None, evaluates all classes).')

    args = parser.parse_args()

    args.classes = list(map(int, args.classes.split(','))) if args.classes else None
    if args.project is None:
        del args.project

    val_kwargs = vars(args)
    evaluator = RTDETREvaluator(**val_kwargs)
    results = evaluator.evaluate_model()
    # print(results)  # Print the evaluation results