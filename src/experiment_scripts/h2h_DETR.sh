### Head-to-Head Experiments Script
### RT-DETR can freeze upto 29 layers
DATA_DIR="/home/data"
WANDB_RUNS_DIR="/home/wandb-runs"

# Real PACE data (total 11323 images)
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 25 --fraction 1.76632 --name real-only-200 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 25 --fraction 4.4158 --name real-only-500 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 25 --fraction 8.8316 --name real-only-1000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 25 --fraction 17.6632 --name real-only-2000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 25 --fraction 44.1579 --name real-only-5000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/pace_v3.yaml --epochs 50 --freeze 25 --fraction 88.3158 --name real-only-10000 --project pace-v3-h2h-detr

python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/real-only-200 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/real-only-500 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/real-only-1000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/real-only-2000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/real-only-5000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/real-only-10000 --split test

# Diffusion
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 25 --fraction 1 --name diffusion-only-200 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 25 --fraction 2.5 --name diffusion-only-500 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 25 --fraction 5 --name diffusion-only-1000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 25 --fraction 10 --name diffusion-only-2000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 25 --fraction 25 --name diffusion-only-5000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 50 --freeze 25 --fraction 50 --name diffusion-only-10000 --project pace-v3-h2h-detr

python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/diffusion-only-200 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/diffusion-only-500 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/diffusion-only-1000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/diffusion-only-2000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/diffusion-only-5000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/diffusion-only-10000 --split test

# # 3D RP
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 25 --fraction 1 --name 3drp-only-200 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 25 --fraction 2.5 --name 3drp-only-500 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 25 --fraction 5 --name 3drp-only-1000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 25 --fraction 10 --name 3drp-only-2000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 25 --fraction 25 --name 3drp-only-5000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 50 --freeze 25 --fraction 50 --name 3drp-only-10000 --project pace-v3-h2h-detr

python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3drp-only-200 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3drp-only-500 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3drp-only-1000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3drp-only-2000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3drp-only-5000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3drp-only-10000 --split test

# # 3D Copy-Paste
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 25 --fraction 1 --name 3dcp-only-200 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 25 --fraction 2.5 --name 3dcp-only-500 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 25 --fraction 5 --name 3dcp-only-1000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 25 --fraction 10 --name 3dcp-only-2000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 25 --fraction 25 --name 3dcp-only-5000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 50 --freeze 25 --fraction 50 --name 3dcp-only-10000 --project pace-v3-h2h-detr

python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3dcp-only-200 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3dcp-only-500 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3dcp-only-1000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3dcp-only-2000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3dcp-only-5000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/3dcp-only-10000 --split test

# # CNP-PACE
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 50 --freeze 25 --fraction 1 --name cnp-only-200 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 50 --freeze 25 --fraction 2.5 --name cnp-only-500 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 50 --freeze 25 --fraction 5 --name cnp-only-1000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 50 --freeze 25 --fraction 10 --name cnp-only-2000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 50 --freeze 25 --fraction 25 --name cnp-only-5000 --project pace-v3-h2h-detr
python src/finetune_RT_DETR.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 50 --freeze 25 --fraction 50 --name cnp-only-10000 --project pace-v3-h2h-detr

python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/cnp-only-200 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/cnp-only-500 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/cnp-only-1000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/cnp-only-2000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/cnp-only-5000 --split test
python src/evaluate_RT_DETR.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-detr/cnp-only-10000 --split test
