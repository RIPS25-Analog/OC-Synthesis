import os
import glob
import yaml
import wandb
from finetune_YOLO import YOLOFinetuner
from evaluate_YOLO import YOLOEvaluator

wandb_prefix = 'vikhyat-3-org/pace-v2/'
root_dir = '/home/wandb-runs/pace-v2/'
orig_project_name, extended_project_name = 'pace-v2', 'pace-v2-extended'
extended_project_dir = root_dir.replace(orig_project_name, extended_project_name)

wandb_api = wandb.Api()
				
## Find the best performing run for each sweep set (e.g. real-only-20), using the mAP50 value from wandb API 
runs_to_extend = []
for sweep_name_dir in glob.glob(os.path.join(root_dir, '*/')):
	# there can be multiple restarted sweeps with the same name, but sweep IDs are unique
	sweep_ids = [x for x in os.listdir(sweep_name_dir) if x!='discarded']
	if len(sweep_ids)!=1:
		print(f"{len(sweep_ids)} sweeps found in {sweep_name_dir}, unsure which to use, so skipping")
		continue

	sweep_id = sweep_ids[0]
	sweep_dir = os.path.join(sweep_name_dir, sweep_id)

	sweep = wandb_api.sweep(wandb_prefix + sweep_id)
	best_run = sorted(sweep.runs, key=lambda run: run.summary.get("mAP50", 0), reverse=True)[0]

	ext_run_name = best_run.sweep.name + '__' + best_run.sweep.id + '__' + best_run.name
	ext_run_path = os.path.join(extended_project_dir, ext_run_name)
	if os.path.exists(ext_run_path):
		val_results_path = os.path.join(extended_project_dir, 'val_' + ext_run_name, 'simple_evaluation_results.yaml')
		assert os.path.exists(val_results_path), f"Extended run dir found but evaluation results missing: {val_results_path}"
			
		print(f"Extended run {ext_run_path} already exists, skipping.")
		continue

	mAP50 = best_run.summary.get("mAP50", None)
	if mAP50 is None:
		print(f"mAP50 not found for only run {best_run.name} in sweep {sweep_id}")
		continue
	
	print(f"Best run found to extend in {sweep_name_dir}: {best_run.name} with mAP50={mAP50}")
	print(f"\t succesfully checked that {ext_run_path} didn't exist")

	args = yaml.safe_load(open(os.path.join(sweep_dir, best_run.name, 'args.yaml')))
	args['epochs'] = 50
	args['patience'] = 20
	args['data'] = args['data'].replace('pace-v2-val.yaml', 'pace_v2.yaml')
	args['name'] = best_run.sweep.name + '__' + best_run.sweep.id + '__' + best_run.name
	args['project'] = extended_project_name
	args['save_dir'] = os.path.join('/home/wandb-runs/') #, extended_project_name, best_run.sweep.name, best_run.sweep.id)
	args['val'] = True
	print(f"\t Save Dir: {args['save_dir']}")

	# ############################### TEMP #######################################################################
	# args['epochs'] = 2
	# args['fraction'] = 0.001	
	runs_to_extend.append((best_run, args))


for run, args in runs_to_extend:
	print(f"\n\nNow working on extending run {args['name']}; being saved to {args['save_dir']}")
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
		'save_dir': args['save_dir']
	}

	evaluator = YOLOEvaluator(**evaluator_args)
	val_results = evaluator.evaluate_model()

	print('Basic metrics:')
	print(val_results.get('metrics'))