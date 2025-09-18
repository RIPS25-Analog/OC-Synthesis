### Fixed hyperparam, varying #real-imgs, experiments script
DATA_DIR="/home/data"
WANDB_RUNS_DIR="/home/wandb-runs"

########### Train on real-only PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name real-only-200 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name real-only-500 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name real-only-1000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name real-only-2000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name real-only-5000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name real-only-10000 --project pace-v3-h2h-yolo-vid-shuf

python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/real-only-200 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/real-only-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/real-only-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/real-only-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/real-only-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/real-only-10000 --split test

########### Pretrain on Synthetic-only data (total 20000 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_v3.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 20 --name cnp-only-20000_rand --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/cnp_bg20k_v3.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 20 --name cnp_bg20k-only-20000_rand --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/diffusion_v3.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --name diffusion-only-20000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_RP_v3.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --name 3DRP-only-20000 --project pace-v3-h2h-yolo-vid-shuf
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/3d_CopyPaste_v3.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --name 3DCP-only-20000 --project pace-v3-h2h-yolo-vid-shuf

python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-only-20000_rand --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-only-20000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-only-20000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-only-20000 --split test

########### Finetuned CNP-pretrained model on real PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name cnp-seq-200 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name cnp-seq-500 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name cnp-seq-1000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name cnp-seq-2000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name cnp-seq-5000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name cnp-seq-10000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-only-20000_rand/weights/last.pt

python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-seq-200 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-seq-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-seq-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-seq-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-seq-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp-seq-10000 --split test

########### Finetuned CNP (BG 20k)-pretrained model on real PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name cnp_bg20k-seq-200 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name cnp_bg20k-seq-500 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name cnp_bg20k-seq-1000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name cnp_bg20k-seq-2000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name cnp_bg20k-seq-5000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name cnp_bg20k-seq-10000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-only-20000_rand/weights/last.pt

python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-seq-200 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-seq-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-seq-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-seq-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-seq-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/cnp_bg20k-seq-10000 --split test

########### Finetuned diffusion-pretrained model on real PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name diffusion-seq-200 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name diffusion-seq-500 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name diffusion-seq-1000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name diffusion-seq-2000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name diffusion-seq-5000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name diffusion-seq-10000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-only-20000/weights/last.pt

python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-seq-200 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-seq-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-seq-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-seq-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-seq-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/diffusion-seq-10000 --split test

########### Finetuned 3dRP-pretrained model on real PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name 3DRP-seq-200 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name 3DRP-seq-500 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name 3DRP-seq-1000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name 3DRP-seq-2000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name 3DRP-seq-5000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name 3DRP-seq-10000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-only-20000/weights/last.pt

python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-seq-200 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-seq-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-seq-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-seq-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-seq-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DRP-seq-10000 --split test

########### Finetuned 3DCP-pretrained model on real PACE data (total 11323 images)
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 1.76632 --name 3DCP-seq-200 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 4.4158 --name 3DCP-seq-500 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 8.8316 --name 3DCP-seq-1000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 17.6632 --name 3DCP-seq-2000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 44.1579 --name 3DCP-seq-5000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-only-20000/weights/last.pt
python src/finetune_YOLO.py --data ${DATA_DIR}/configs/pace_v3_video-shuffled.yaml --epochs 25 --lr0 0.001 --freeze 20 --imgsz 960 --fraction 88.3158 --name 3DCP-seq-10000 --project pace-v3-h2h-yolo-vid-shuf --model ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-only-20000/weights/last.pt

python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-seq-200 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-seq-500 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-seq-1000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-seq-2000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-seq-5000 --split test
python src/evaluate_YOLO.py --run ${WANDB_RUNS_DIR}/pace-v3-h2h-yolo-vid-shuf/3DCP-seq-10000 --split test