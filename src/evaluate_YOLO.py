from ultralytics import YOLO
import os
import yaml
import argparse

class YOLOevaluator:
    def __init__(self, run_path=None, data_path=None, data_split='val', model=None):
        self.run_path = run_path

        if model is None:
            assert run_path is not None, "If no model is provided, run_path must be specified."
            self.model_path = os.path.join(run_path, 'weights', 'best.pt')
            self.model = YOLO(self.model_path, task='detect')
            self.data_path = data_path if data_path else self._get_data_path_from_model_yaml()
        else:
            self.model = model
            assert data_path is not None, "If a model is provided, data_path must also be specified."
            self.data_path = data_path

        print('Model info:', self.model.info())
        print('Evaluating on dataset:', self.data_path)
        
        self.data_split = data_split

    def _get_data_path_from_model_yaml(self):
        """Fetch data path from the model's YAML configuration if not provided."""
        with open(os.path.join(self.run_path, 'args.yaml'), 'r') as file:
            yaml_content = file.read()
            yaml_data = yaml.safe_load(yaml_content)
            data_path = yaml_data.get('data', None)
            
        print(f"Data path: {data_path}")
        assert data_path, ValueError("Data path not found in the run's YAML config file.")
        return data_path

    def evaluate_model(self):
        """Evaluate the YOLO model and return results."""
        project_name = 'runs/' + self.data_path.split('/')[-1].split('.')[0]
        results = self.model.val(data=self.data_path, project=project_name, split=self.data_split)
        print(f"Evaluating on dataset: {self.data_path} with split: {self.data_split}")
        
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
                        'model_path': self.model_path if hasattr(self, 'model_path') else None,
                        'data_path': self.data_path
                        }

        # Save the results to a file
        results_file = os.path.join(results.save_dir, 'evaluation_results.yaml')
        with open(results_file, 'w') as file:
            yaml.dump(result_dict, file)
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a YOLO model with specified parameters.')
    parser.add_argument('--run_path', type=str, default='yolo11n.pt', help='Path to the YOLO run directory (to fetch best weights from).')
    parser.add_argument('--data_path', type=str, required=False, help='Path to the dataset configuration file (YAML). (Optional, will be fetched from the run YAML if not provided).')
    parser.add_argument('--data_split', type=str, default='val', choices=['train', 'val', 'test'], help='Specify the dataset split to evaluate on (default: val).')

    args = parser.parse_args()

    evaluator = YOLOevaluator(args.run_path, args.data_path, args.data_split)
    results = evaluator.evaluate_model()
    print(results)  # Print the evaluation results