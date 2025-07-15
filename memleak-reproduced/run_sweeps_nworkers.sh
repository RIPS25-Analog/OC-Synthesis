# run hyperparam_opt_YOLO.py
trials=(1 2 3 4 5 6 7 8)

# Loop through each trial
for trial_num in "${trials[@]}"; do
    echo "Run number $trial_num"
    python finetune_YOLO.py --data_path /home/data/cnp-v0-15365.yaml --epochs 1 --batch 32 --workers 8
    echo "Completed run $trial_num."
done