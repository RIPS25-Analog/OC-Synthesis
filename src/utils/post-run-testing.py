import os
import argparse
import glob
import yaml
import wandb

import sys
sys.path.append('src')

from evaluate_YOLO import YOLOEvaluator

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run test-set evaluation on all runs within a project',
									  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--project', type=str, required=True, help='Project name')
	args = parser.parse_args()

	wandb_runs_dir = '/home/wandb-runs'

	project_dir = f'{wandb_runs_dir}/{args.project}/'
					
	## Find all runs which haven't been evaluated yet
	runs_to_evaluate = []
	print(f"Finding all runs to evaluate in {project_dir}")
	for run_dir in glob.glob(os.path.join(project_dir, '*/')):
		full_name = run_dir.split('/')[-2]
		val_dirs = glob.glob(os.path.join(run_dir, 'val*/'))

		if val_dirs:
			val_results_path = os.path.join(val_dirs[0], 'simple_evaluation_results.yaml')
			assert os.path.exists(val_results_path), f"Eval run dir found but evaluation results missing: {val_results_path}"
			print(f"Validation results {val_results_path} already exists, skipping.")
			continue

		train_args = yaml.safe_load(open(os.path.join(run_dir, 'args.yaml')))
		train_args['name'] = full_name
		if not train_args['save_dir'].startswith(wandb_runs_dir):
			train_args['save_dir'] = os.path.join(wandb_runs_dir, train_args['save_dir'])
		print(f"\t Save Dir: {train_args['save_dir']}")

		runs_to_evaluate.append((run_dir, train_args))

	# Evaluate runs on test set
	for run_dir, train_args in runs_to_evaluate:
		print(f"\n\nNow evaluating run {train_args['name']}; being saved to {train_args['save_dir']}")
		evaluator_args = {
			'run': str(train_args['save_dir']),
			'batch': 32,
			'imgsz': train_args['imgsz'],
			'project': train_args['project'],
			'split': 'test',  # Evaluate on the final test set
			'save_dir': train_args['save_dir']
		}

		evaluator = YOLOEvaluator(**evaluator_args)
		val_results = evaluator.evaluate_model()

		print('Basic metrics:')
		print(val_results.get('metrics'))