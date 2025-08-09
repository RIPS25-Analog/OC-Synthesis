###### YOLOWorld on PACE real data only
##### for fraction in 10 20 50 100; do
#####     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name YOLOWorld-real-only-$fraction --model yolov8s-worldv2.pt
##### done

## YOLO11 on PACE real data only
# for fraction in 10 20 50 100; do
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name real-only-$fraction
# done

## YOLO11 on 2D CnP PACE synthetic data only
# for fraction in 10 20 50 100; do
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/cnp_pace_v2-pace_eval.yaml --project pace-v2 --sweep_name 2D_CNP-only-$fraction
# done

# YOLO11 on 3D render synthetic data only
for fraction in 10 20 50 100; do
    python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/3d_pace_v0-pace_eval.yaml --project pace-v2 --sweep_name 3D_render-only-$fraction
done

# python src/post-sweep-testing.py 