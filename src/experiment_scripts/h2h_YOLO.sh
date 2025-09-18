### Head-to-Head Experiments Script
DATA_DIR="/home/data"
WANDB_RUNS_DIR="/home/wandb-runs"

# Real data
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 1.76632 --name real-only-250 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 4.4158 --name real-only-500 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 8.8316 --name real-only-1000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 17.6632 --name real-only-2000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 44.1579 --name real-only-5000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 88.3158 --name real-only-10000 --project pace-v3 --val False
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/real-only-250 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/real-only-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/real-only-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/real-only-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/real-only-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/real-only-10000 --split test

# CNP-PACE
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp-pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 1.25 --name cnp-pace-250 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp-pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 2.5 --name cnp-pace-500 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp-pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 5 --name cnp-pace-1000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp-pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 10 --name cnp-pace-2000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp-pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 25 --name cnp-pace-5000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp-pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 50 --name cnp-pace-10000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp-pace_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 100 --name cnp-pace-20000 --project pace-v3 --val False
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/cnp-pace-250 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/cnp-pace-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/cnp-pace-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/cnp-pace-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/cnp-pace-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/cnp-pace-10000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/cnp-pace-20000 --split test

# Diffusion
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 1.25 --name diffusion-250 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 2.5 --name diffusion-500 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 5 --name diffusion-1000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 10 --name diffusion-2000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 25 --name diffusion-5000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 50 --name diffusion-10000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 100 --name diffusion-20000 --project pace-v3 --val False
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/diffusion-250 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/diffusion-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/diffusion-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/diffusion-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/diffusion-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/diffusion-10000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/diffusion-20000 --split test

# 3D RP
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 1.25 --name 3D-RP-250 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 2.5 --name 3D-RP-500 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 5 --name 3D-RP-1000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 10 --name 3D-RP-2000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 25 --name 3D-RP-5000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 50 --name 3D-RP-10000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 100 --name 3D-RP-20000 --project pace-v3 --val False
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-RP-250 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-RP-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-RP-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-RP-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-RP-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-RP-10000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-RP-20000 --split test

# 3D Copy-Paste
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 1.25 --name 3D-CopyPaste-250 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 2.5 --name 3D-CopyPaste-500 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 5 --name 3D-CopyPaste-1000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 10 --name 3D-CopyPaste-2000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 25 --name 3D-CopyPaste-5000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 50 --name 3D-CopyPaste-10000 --project pace-v3 --val False
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 20 --lr0 0.0003 --fraction 100 --name 3D-CopyPaste-20000 --project pace-v3 --val False
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-CopyPaste-250 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-CopyPaste-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-CopyPaste-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-CopyPaste-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-CopyPaste-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-CopyPaste-10000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3/3D-CopyPaste-20000 --split test