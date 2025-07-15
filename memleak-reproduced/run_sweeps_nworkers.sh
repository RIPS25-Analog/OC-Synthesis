# run hyperparam_opt_YOLO.py with different numbers of workers each time
WORKERS=(0)

# Loop through each number of workers
for nworkers in "${WORKERS[@]}"; do
    echo "Running with $nworkers workers..."
    python hyperparam_opt_YOLO.py --nworkers "$nworkers"
    echo "Completed run with $nworkers workers."
done