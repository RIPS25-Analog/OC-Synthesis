from ultralytics import YOLO
import argparse

class YOLOfinetuner:
    def __init__(self, model_path, data_path, epochs, freeze):
        self.model = YOLO(model_path, task='detect')
        print(self.model.info())
        self.data_path = data_path
        self.epochs = epochs
        self.freeze = freeze

    def train_model(self):
        project_name = 'runs/' + self.data_path.split('/')[-1].split('.')[0]
        results = self.model.train(data=self.data_path, epochs=self.epochs, freeze=self.freeze,
                                    project=project_name)
        return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a YOLO model with specified parameters.')
    parser.add_argument('--model_path', type=str, default='yolo11n.pt', help='Path to the YOLO model file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset configuration file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--freeze', type=int, default=23, help='Number of layers to freeze during training.')

    args = parser.parse_args()

    finetuner = YOLOfinetuner(args.model_path, args.data_path, args.epochs, args.freeze)
    results = finetuner.train_model()
    print(results)  # Print the training results