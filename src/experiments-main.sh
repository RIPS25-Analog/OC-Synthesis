# #### Generate PACE V3 data
# # python src/Cut-and-Paste/dataset_generator.py --n_images 20000 /home/data/pace/pace_v3-foreground_objects/ /home/data/processed/cnp-pace/pace_v3

# #### testing hyperparam
# python src/hyperparam_opt_YOLO.py --sweep_count 3 --data /home/data/configs/3d_RP_v3.yaml --fraction 0.05 --project TMP_pace-v3-main-yolo11n --sweep_name 3d_RP-only
# python src/hyperparam_opt_YOLO.py --sweep_count 5 --data /home/data/configs/pace_v3.yaml --project TMP_pace-v3-main-yolo11n --sweep_name 3d_RP-seq --parent_sweep_name_dir /home/wandb-runs/TMP_pace-v3-main-yolo11n/3d_RP-only

########### Real only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name real-only

########### Synthetic only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/cnp_pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name 2D_CNP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/diffusion_v3.yaml --project pace-v3-main-yolo11n --sweep_name diffusion-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/3d_RP_v3.yaml --project pace-v3-main-yolo11n --sweep_name 3D_RP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/3d_CopyPaste_v3.yaml --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-only

########### Mixed
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/cnp_pace_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name 2D_CNP-mixed
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/diffusion_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name diffusion-mixed
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/3d_RP_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name 3D_RP-mixed
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/3d_CopyPaste_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-mixed

########### Sequential
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name 2D_CNP-seq --parent_sweep_name_dir /home/wandb-runs/pace-v3-main-yolo11n/2D_CNP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name diffusion-seq --parent_sweep_name_dir /home/wandb-runs/pace-v3-main-yolo11n/diffusion-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name 3D_RP-seq --parent_sweep_name_dir /home/wandb-runs/pace-v3-main-yolo11n/3D_RP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data /home/data/configs/pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-seq --parent_sweep_name_dir /home/wandb-runs/pace-v3-main-yolo11n/3D_CopyPaste-only

python src/post-sweep-testing.py --project pace-v3-main-yolo11n

