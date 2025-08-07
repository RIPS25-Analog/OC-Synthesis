for fraction in 0.1 0.2 0.5 1.0; do
    frac_full=${fraction/0./}
    if [ "$fraction" = "1.0" ]; then
        frac_full=100
    else
        frac_full=${frac_full}0
    fi
    python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/pace-v2-val.yaml --project pace-v2 --sweep_name real-only-$frac_full
done

for fraction in 0.1 0.2 0.5 1.0; do
    frac_full=${fraction/0./}
    if [ "$fraction" = "1.0" ]; then
        frac_full=100
    else
        frac_full=${frac_full}0
    fi
    python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/cnp_pace_v2-pace_eval.yaml --project pace-v2 --sweep_name 2D_CNP-only-$frac_full
done