### Head-to-Head Experiments Script
DATA_DIR="/home/data"
WANDB_RUNS_DIR="/home/wandb-runs"

# Real data
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 1.76632 --name real-only-200 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 4.4158 --name real-only-500 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 8.8316 --name real-only-1000 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 17.6632 --name real-only-2000 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 44.1579 --name real-only-5000 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 88.3158 --name real-only-10000 --project pace-v3-h2h-yolo-vid-shuf

# CNP-PACE
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 1 --name cnp-only-200 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 2.5 --name cnp-only-500 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 5 --name cnp-only-1000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 10 --name cnp-only-2000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 25 --name cnp-only-5000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 50 --name cnp-only-10000 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 100 --name cnp-only-20000 --project pace-v3-h2h-yolo-vid-shuf

# Diffusion
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 1 --name diffusion-only-200 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 2.5 --name diffusion-only-500 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 5 --name diffusion-only-1000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 10 --name diffusion-only-2000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 25 --name diffusion-only-5000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 50 --name diffusion-only-10000 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 100 --name diffusion-only-20000 --project pace-v3-h2h-yolo-vid-shuf

# 3D RP
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 1 --name 3DRP-only-200 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 2.5 --name 3DRP-only-500 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 5 --name 3DRP-only-1000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 10 --name 3DRP-only-2000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 25 --name 3DRP-only-5000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 50 --name 3DRP-only-10000 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 100 --name 3DRP-only-20000 --project pace-v3-h2h-yolo-vid-shuf

# 3D Copy-Paste
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 1 --name 3DCP-only-200 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 2.5 --name 3DCP-only-500 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 5 --name 3DCP-only-1000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 10 --name 3DCP-only-2000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 25 --name 3DCP-only-5000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 50 --name 3DCP-only-10000 --project pace-v3-h2h-yolo-vid-shuf
# python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --freeze 20 --lr0 0.001 --imgsz 960 --fraction 100 --name 3DCP-only-20000 --project pace-v3-h2h-yolo-vid-shuf

# Post-run testing to evaluate all runs
python src/utils/post-run-testing.py --project pace-v3-h2h-yolo-vid-shuf