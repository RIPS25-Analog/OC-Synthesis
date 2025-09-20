########### CNP (BG 20k)-only pretrained model
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_bg20k_v3.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 20 --name cnp_bg20k-only-20000_rand --project pace-v3-h2h-yolo-vid-shuf
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand --split test

########### Finetuned CNP (BG 20k)-pretrained model on real PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name cnp_bg20k-seq-200 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name cnp_bg20k-seq-500 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name cnp_bg20k-seq-1000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name cnp_bg20k-seq-2000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name cnp_bg20k-seq-5000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name cnp_bg20k-seq-10000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt

########### Diffusion (BG 20k)-only pretrained model
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_bg20k_v3.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 20 --name diffusion_bg20k-only-20000 --project pace-v3-h2h-yolo-vid-shuf
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion_bg20k-only-20000 --split test

########### Finetuned Diffusion (BG 20k)-pretrained model on real PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name diffusion_bg20k-seq-200 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion_bg20k-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name diffusion_bg20k-seq-500 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion_bg20k-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name diffusion_bg20k-seq-1000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion_bg20k-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name diffusion_bg20k-seq-2000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion_bg20k-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name diffusion_bg20k-seq-5000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion_bg20k-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name diffusion_bg20k-seq-10000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion_bg20k-only-20000/weights/last.pt

# Post-run testing to evaluate all runs
python src/utils/post-run-testing.py --project pace-v3-h2h-yolo-vid-shuf