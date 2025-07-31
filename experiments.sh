# python src/hyperparam_opt_YOLO.py --epochs 20 --sweep_count 25 --data /home/data/mech-94-val.yaml --project cnp-mech-94-val --sweep_name real-only
# python src/hyperparam_opt_YOLO.py --epochs 20 --sweep_count 25 --data /home/data/cnp-v0-4985-mech-94-val.yaml --project all-mech-94-val --sweep_name real-only
# python src/hyperparam_opt_YOLO.py --epochs 20 --sweep_count 25 --data /home/data/cnp-v0-4985-mech-val.yaml --project all-mech-94-val
# # python src/hyperparam_opt_YOLO.py --epochs 20 --sweep_count 25 --data /home/data/mech-94-val.yaml --project all-mech-94-val --model /home/wandb-runs/cnp-v0-4985-mech-val/p5e9i457/train13/weights/best.pt

python src/hyperparam_opt_YOLO.py --epochs 3 --sweep_count 15 --data /home/data/configs/mech-94-val.yaml --project all-mech-94-val --sweep_name 75k-real-only
python src/hyperparam_opt_YOLO.py --epochs 3 --sweep_count 15 --data /home/data/configs/cnp-v0-75965-mech-94-val.yaml --project all-mech-94-val --sweep_name 75k-mixed
python src/hyperparam_opt_YOLO.py --epochs 3 --sweep_count 15 --data /home/data/configs/cnp-v0-75965-mech-val.yaml --project all-mech-94-val --sweep_name 75k-synthetic-only
# python src/hyperparam_opt_YOLO.py --epochs 20 --sweep_count 25 --data /home/data/mech-94-val.yaml --project all-mech-94-val --model /home/wandb-runs/cnp-v0-4985-mech-val/p5e9i457/train13/weights/best.pt