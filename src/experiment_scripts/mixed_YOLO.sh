## Mixed runs

list=("configs/mixed_synthetic_data/3dRP.yaml" "configs/mixed_synthetic_data/cnp-3dCP.yaml" "configs/mixed_synthetic_data/cnp-3dRP.yaml" "configs/mixed_synthetic_data/cnp.yaml" "configs/mixed_synthetic_data/diffusion-3dCP.yaml" "configs/mixed_synthetic_data/diffusion-3dRP.yaml" "configs/mixed_synthetic_data/diffusion-cnp.yaml" "configs/mixed_synthetic_data/diffusion-real.yaml" "configs/mixed_synthetic_data/diffusion.yaml" "configs/mixed_synthetic_data/real-3dCP.yaml" "configs/mixed_synthetic_data/real-3dRP.yaml" "configs/mixed_synthetic_data/real-cnp.yaml" "configs/mixed_synthetic_data/real.yaml" "configs/mixed_synthetic_data/3dCP.yaml" "configs/mixed_synthetic_data/3dRP-3dCP.yaml")
for i in "${list[@]}"; do
    echo "Processing $i file..."
    name=$(echo $i | cut -d'/' -f3 | cut -d'.' -f1)
    python src/finetune_YOLO.py --data /home/data/$i --epochs 50 --freeze 20 --lr0 0.0003 --name mixed-$name --project pace-v3-mixing --val False
    python src/evaluate_YOLO.py --run /home/wandb-runs/pace-v3-mixing/mixed-$name --project /home/wandb-runs/pace-v3-mixing/mixed-$name --split test
done
