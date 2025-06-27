import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from src.training.trainer import YOLOTrainer
from src.data.data_manager import DataManager
from src.evaluation.evaluator import Evaluator
from src.utils.experiment_logger import ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description='Run YOLO11 experiment')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to experiment config file')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(f"experiments/{exp_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger(exp_dir, config)
    
    try:
        # Data preparation
        data_manager = DataManager(config)
        train_loader, val_loader = data_manager.prepare_data()
        
        # Training
        trainer = YOLOTrainer(config, exp_dir)
        model = trainer.train(train_loader, val_loader)
        
        # Evaluation
        evaluator = Evaluator(config, exp_dir)
        results = evaluator.evaluate(model, val_loader)
        
        # Log results
        logger.log_results(results)
        
        print(f"Experiment completed. Results saved to {exp_dir}")
        
    except Exception as e:
        logger.log_error(str(e))
        raise

if __name__ == "__main__":
    main()
