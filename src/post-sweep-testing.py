import os
import glob
import shutil
import csv
import yaml
import re
from finetune_YOLO import YOLOFinetuner
from evaluate_YOLO import YOLOEvaluator
root_dir = '/home/wandb-runs/pace-v2/'

## Check which run within a given sweep set is missing a val dir which contains a yaml referring to it
## to do this, first go through all the val dirs in that sweep set and check their yaml file to see which run they refer to
for sweep_name_dir in sorted(glob.glob(os.path.join(root_dir, '*-*-*/'))):
	sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
	if len(sweep_ids)!=1:
		print(f"{len(sweep_ids)} sweeps found in {sweep_name_dir}, unsure which to use, so skipping")
		continue
	sweep_id = sweep_ids[0]
	sweep_dir = os.path.join(sweep_name_dir, sweep_id)

	runs_w_validation = set() # runs which have corresponding validation folders present
	val_dirs = [d for d in os.listdir(sweep_dir) if re.match(r'^val\d+', d)]
	for val_dir in val_dirs:
		yaml_file = os.path.join(sweep_dir, val_dir, 'simple_evaluation_results.yaml')
		if os.path.exists(yaml_file):
			with open(yaml_file, 'r') as f:
				config = yaml.safe_load(f)
				run_name = config.get('model').split('/')[-3]
				if run_name:
					runs_w_validation.add(run_name)

	all_runs = set(x.split('/')[-1] for x in glob.glob(os.path.join(sweep_dir, '*-*-*')) if 'extended' not in x)
	print(sweep_dir)
	print(f'- Found {len(val_dirs)} val directories and {len(all_runs)} runs')

	uncovered_runs = all_runs - runs_w_validation
	if uncovered_runs:
		print(f"- Runs without val directories:")
	for run in uncovered_runs:
		print(f"	- {run}")
				
## Find the k best performing runs for a given sweep set directory
## use the metrics/mAP50(B) value from the metrics entry of the simple_evaluation_results.yaml file in the val dirs
top_k = 1
best_runs_for_set = dict()
for sweep_name_dir in glob.glob(os.path.join(root_dir, '*-*-*/')):
	sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
	if len(sweep_ids)!=1:
		print(f"{len(sweep_ids)} sweeps found in {sweep_dir}, unsure which to use, so skipping")
		continue
	sweep_id = sweep_ids[0]
	sweep_dir = os.path.join(sweep_name_dir, sweep_id)

	map50_scores = dict()  # run_name -> mAP50 score
	val_dirs = [d for d in os.listdir(sweep_dir) if re.match(r'^val\d+', d)]
	for val_dir in val_dirs:
		yaml_file = os.path.join(sweep_dir, val_dir, 'simple_evaluation_results.yaml')
		if os.path.exists(yaml_file):
			with open(yaml_file, 'r') as f:
				config = yaml.safe_load(f)
				mAP50 = config.get('metrics', {}).get('metrics/mAP50(B)', 0)
				run_name = config.get('model').split('/')[-3]
				map50_scores[run_name] = mAP50
				
	sweep_set_id = sweep_dir.split('/')[-1]
	best_runs_for_set[sweep_set_id] = sorted(map50_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
	print(f"Best runs in set {sweep_dir}:")
	for run, score in best_runs_for_set[sweep_set_id]	:
		print(f" - {run}: {score*100:.2f}%") # print percentage to 2 decimal place

for sweep_name_dir in glob.glob(os.path.join(root_dir, '*-*-*/')):
	sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
	if len(sweep_ids)!=1:
		print(f"{len(sweep_ids)} sweeps found in {sweep_name_dir}, unsure which to use, so skipping")
		continue
	sweep_id = sweep_ids[0]
	sweep_dir = os.path.join(sweep_name_dir, sweep_id)
	best_runs = best_runs_for_set[sweep_dir.split('/')[-1]]
	
	for run, old_mAP in best_runs:
		print(f"Re-finetuning {run} with old mAP {old_mAP*100:.2f}%")

		args = yaml.safe_load(open(os.path.join(sweep_dir, run, 'args.yaml')))
		args['epochs'] = 50
		args['patience'] = 20
		args['name'] = args['name'] + '_extended'
		args['save_dir'] = os.path.join(sweep_dir, args['name'])
		args['data'] = args['data'].replace('pace-v2-val.yaml', 'pace-v2.yaml')
		args['project'] = sweep_dir.replace('pace-v2/', 'pace-v2-extended/')
		args['val'] = True

		full_run_name = os.path.join(args['project'], args['name'])
		if os.path.exists(full_run_name):
			print(f"Project directory {full_run_name} in {args['project']} already exists, skipping {run}")
			continue
		
		# ############################### TEMP #######################################################################
		# args['epochs'] = 2
		# args['fraction'] = 0.001

		yolo_finetuner = YOLOFinetuner(**args)
		results_train = yolo_finetuner.train_model()
		
		evaluator_args = {
			'run': str(results_train.save_dir),
			'batch': 32,
			# 'imgsz': config.get('eval_imgsz', 640),
			'project': args['project'],
			'split': 'test',  # Evaluate on the final test set
			'name': 'val_'+args['name'],
		}
		evaluator = YOLOEvaluator(**evaluator_args)
		val_results = evaluator.evaluate_model()

		print('Basic metrics:')
		print(val_results.get('metrics'))