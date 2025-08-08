for fraction in 10 20 50 100; do
    python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name YOLOWorld-real-only-$fraction --model yolov8s-worldv2.pt
done

# for fraction in 0.1 0.2 0.5 1.0; do
#     frac_full=${fraction/0./}
#     if [ "$fraction" = "1.0" ]; then
#         frac_full=100
#     else
#         frac_full=${frac_full}0
#     fi
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/cnp_pace_v2-pace_eval.yaml --project pace-v2 --sweep_name 2D_CNP-only-$frac_full
# done

# #### DO the real-only 20% mix since 10%-50%-100% weren't enough information
# for fraction in 0.2; do
#     frac_full=${fraction/0./}
#     if [ "$fraction" = "1.0" ]; then
#         frac_full=100
#     else
#         frac_full=${frac_full}0
#     fi
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/pace-v2-val.yaml --project pace-v2 --sweep_name real-only-$frac_full
# done