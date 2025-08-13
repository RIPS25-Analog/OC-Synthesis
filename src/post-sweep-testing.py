import os
import glob
import yaml
import wandb
from finetune_YOLO import YOLOFinetuner
from evaluate_YOLO import YOLOEvaluator

root_dir = '/home/wandb-runs/pace-v2/'
orig_project_name, extended_project_name = 'pace-v2', 'pace-v2-extended'
extended_project_dir = root_dir.replace(orig_project_name, extended_project_name)
wandb_prefix = 'vikhyat-3-org/pace-v2/'

wandb_api = wandb.Api()
				
## Find the best performing run for a given sweep set directory
## using the mAP50 value from the wandb API sweep 
runs_to_extend = []
for sweep_name_dir in glob.glob(os.path.join(root_dir, '*/')):
	# there can be multiple restarted sweeps with the same name, e.g: diffusion-only-10
	sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
	if len(sweep_ids)!=1:
		print(f"{len(sweep_ids)} sweeps found in {sweep_name_dir}, unsure which to use, so skipping")
		continue
	sweep_id = sweep_ids[0]
	sweep_dir = os.path.join(sweep_name_dir, sweep_id)

	sweep = wandb_api.sweep(wandb_prefix + sweep_id)
	best_run = sorted(sweep.runs, key=lambda run: run.summary.get("mAP50", 0), reverse=True)[0]

	full_run_name = os.path.join(extended_project_dir, best_run.sweep.name, best_run.sweep.id, best_run.sweep.name+'_extended')
	if os.path.exists(full_run_name):
		print(f"Extended run {full_run_name} already exists, skipping.")
		continue

	mAP50 = best_run.summary.get("mAP50", None)
	print(f"Best run found to extend in {sweep_name_dir}: {best_run.name} with mAP50={mAP50}")

	args = yaml.safe_load(open(os.path.join(sweep_dir, best_run.name, 'args.yaml')))
	args['epochs'] = 50
	args['patience'] = 20
	args['name'] = args['name'] + '_extended'
	args['save_dir'] = os.path.join(sweep_dir.replace(orig_project_name, extended_project_name), args['name'])
	args['data'] = args['data'].replace('pace-v2-val.yaml', 'pace-v2.yaml')
	args['project'] = sweep_dir.replace(orig_project_name, extended_project_name)
	args['val'] = True

	# ############################### TEMP #######################################################################
	# args['epochs'] = 2
	# args['fraction'] = 0.001	
	runs_to_extend.append((best_run, args))

for run, args in runs_to_extend:				
		yolo_finetuner = YOLOFinetuner(**args)
		results_train = yolo_finetuner.train_model()

		run_name = run
		eval_imgsz = run.config['eval_imgsz']
		evaluator_args = {
			'run': str(results_train.save_dir),
			'batch': 32,
			'imgsz': eval_imgsz,
			'project': args['project'],
			'split': 'test',  # Evaluate on the final test set
			'name': 'val_'+args['name'],
		}

		evaluator = YOLOEvaluator(**evaluator_args)
		val_results = evaluator.evaluate_model()

		print('Basic metrics:')
		print(val_results.get('metrics'))