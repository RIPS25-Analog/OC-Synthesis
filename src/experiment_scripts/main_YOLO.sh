# Define directory variables
DATA_DIR="/home/data"
WANDB_RUNS_DIR="/home/wandb-runs"

# #### Generate PACE V3 data
# python src/Cut-and-Paste/dataset_generator.py --n_images 20000 ${DATA_DIR}/pace/pace_v3-foreground_objects/ ${DATA_DIR}/processed/cnp-pace/pace_v3

########### Real only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name real-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 2.5 --project pace-v3-main-yolo11n --sweep_name real-only-2.5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 5 --project pace-v3-main-yolo11n --sweep_name real-only-5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 10 --project pace-v3-main-yolo11n --sweep_name real-only-10

# ########### Synthetic only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/cnp_pace_v3.yaml --project pace-v3-main-yolo11n --sweep_name 2D_CNP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/diffusion_v3.yaml --project pace-v3-main-yolo11n --sweep_name diffusion-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_RP_v3.yaml --project pace-v3-main-yolo11n --sweep_name 3D_RP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-only

########### Sequential (with x% real data)
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 2.5 --project pace-v3-main-yolo11n --sweep_name 2D_CNP-seq-2.5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/2D_CNP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 2.5 --project pace-v3-main-yolo11n --sweep_name diffusion-seq-2.5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/diffusion-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 2.5 --project pace-v3-main-yolo11n --sweep_name 3D_RP-seq-2.5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/3D_RP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 2.5 --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-seq-2.5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/3D_CopyPaste-only

python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 5 --project pace-v3-main-yolo11n --sweep_name 2D_CNP-seq-5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/2D_CNP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 5 --project pace-v3-main-yolo11n --sweep_name diffusion-seq-5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/diffusion-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 5 --project pace-v3-main-yolo11n --sweep_name 3D_RP-seq-5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/3D_RP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 5 --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-seq-5 --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/3D_CopyPaste-only

python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 10 --project pace-v3-main-yolo11n --sweep_name 2D_CNP-seq --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/2D_CNP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 10 --project pace-v3-main-yolo11n --sweep_name diffusion-seq --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/diffusion-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 10 --project pace-v3-main-yolo11n --sweep_name 3D_RP-seq --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/3D_RP-only
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/pace_v3.yaml --fraction 10 --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-seq --parent_sweep_name_dir ${WANDB_RUNS_DIR}/pace-v3-main-yolo11n/3D_CopyPaste-only

########### Mixed (with x% real data)
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/cnp_pace_v3-mixed-2.5.yaml --project pace-v3-main-yolo11n --sweep_name 2D_CNP-mixed-2.5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/cnp_pace_v3-mixed-5.yaml --project pace-v3-main-yolo11n --sweep_name 2D_CNP-mixed-5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/cnp_pace_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name 2D_CNP-mixed-10

python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/diffusion_v3-mixed-2.5.yaml --project pace-v3-main-yolo11n --sweep_name diffusion-mixed-2.5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/diffusion_v3-mixed-5.yaml --project pace-v3-main-yolo11n --sweep_name diffusion-mixed-5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/diffusion_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name diffusion-mixed-10

python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_RP_v3-mixed-5.yaml --project pace-v3-main-yolo11n --sweep_name 3D_RP-mixed-5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_RP_v3-mixed-2.5.yaml --project pace-v3-main-yolo11n --sweep_name 3D_RP-mixed-2.5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_RP_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name 3D_RP-mixed-10

python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_CopyPaste_v3-mixed-2.5.yaml --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-mixed-2.5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_CopyPaste_v3-mixed-5.yaml --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-mixed-5
python src/hyperparam_opt_YOLO.py --sweep_count 18 --data ${DATA_DIR}/configs/3d_CopyPaste_v3-mixed.yaml --project pace-v3-main-yolo11n --sweep_name 3D_CopyPaste-mixed-10

########### Test all models on test set
python src/utils/post-sweep-testing.py --project pace-v3-main-yolo11n