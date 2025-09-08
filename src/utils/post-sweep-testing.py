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

	wandb_prefix = f'vikhyat-3-org'
	wandb_runs_dir = '/home/wandb-runs'

	project_dir = f'{wandb_runs_dir}/{args.project}/'
	wandb_api = wandb.Api()
					
	## Find the best performing run for each sweep set (e.g. real-only-20), using the mAP50 value from wandb API
	runs_to_evaluate = []
	print(f"Finding best runs to evaluate in {project_dir}")
	for sweep_name_dir in glob.glob(os.path.join(project_dir, '*/')):
		# there can be multiple restarted sweeps with the same name, but sweep IDs are unique
		sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
		if len(sweep_ids)!=1:
			print(f"{len(sweep_ids)} sweeps found in {sweep_name_dir}, unsure which to use, so skipping")
			continue

		sweep_id = sweep_ids[0]
		sweep_dir = os.path.join(sweep_name_dir, sweep_id)

		sweep = wandb_api.sweep(f'{wandb_prefix}/{args.project}/{sweep_id}')
		best_run = sorted(sweep.runs, key=lambda run: run.summary.get("mAP50", 0), reverse=True)[0]

		full_name = best_run.name
		eval_run_name = 'test_' + full_name
		eval_run_path = os.path.join(sweep_dir, eval_run_name)
		if os.path.exists(eval_run_path):
			val_results_path = os.path.join(sweep_dir, eval_run_name, 'simple_evaluation_results.yaml')
			assert os.path.exists(val_results_path), f"Eval run dir found but evaluation results missing: {val_results_path}"

			print(f"Eval run {eval_run_path} already exists, skipping.")
			continue

		mAP50 = best_run.summary.get("mAP50", None)
		if mAP50 is None:
			print(f"mAP50 not found for only run {best_run.name} in sweep {sweep_id}")
			continue
		
		print(f"Best run found to evaluate in {sweep_name_dir}: {best_run.name} with mAP50={mAP50}")
		print(f"\t succesfully checked that {eval_run_path} didn't exist")

		train_args = yaml.safe_load(open(os.path.join(sweep_dir, best_run.name, 'args.yaml')))
		train_args['name'] = full_name
		# train_args['save_dir'] = os.path.join(wandb_runs_dir) #, extended_project_name, best_run.sweep.name, best_run.sweep.id)
		print(f"\t Save Dir: {train_args['save_dir']}")

		runs_to_evaluate.append((best_run, train_args))

	## Run extended finetuning on the best run per sweep (found above)
	for run, train_args in runs_to_evaluate:
		print(f"\n\nNow evaluating run {train_args['name']}; being saved to {train_args['save_dir']}")
		evaluator_args = {
			'run': str(train_args['save_dir']),
			'batch': 32,
			'imgsz': train_args['imgsz'],
			'project': train_args['project'],
			'split': 'test',  # Evaluate on the final test set
			'name': 'test_' + train_args['name'],
			'save_dir': train_args['save_dir']
		}

		evaluator = YOLOEvaluator(**evaluator_args)
		val_results = evaluator.evaluate_model()

		print('Basic metrics:')
		print(val_results.get('metrics'))