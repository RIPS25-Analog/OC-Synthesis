###### YOLOWorld on PACE real data only
##### for fraction in 10 20 50 100; do
#####     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name YOLOWorld-real-only-$fraction --model yolov8s-worldv2.pt
##### done

########### Real only
## YOLO11 on PACE real data only
# for fraction in 10 20 50 100; do
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name real-only-$fraction
# done

########### Synthetic only

## YOLO11 on 2D CnP PACE synthetic data only
# for fraction in 10 20 50 100; do
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/cnp_pace_v2-pace_eval.yaml --project pace-v2 --sweep_name 2D_CNP-only-$fraction
# done

# # YOLO11 on Diffusion synthetic data only
# for fraction in 10 20 50 100; do
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/diffusion_v0-pace_eval.yaml --project pace-v2 --sweep_name diffusion-only-$fraction
# done

# # YOLO11 on 3D render synthetic data only
# for fraction in 10 20 50 100; do
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/3d_pace_v0-pace_eval.yaml --project pace-v2 --sweep_name 3D_render-only-$fraction
# done

# # YOLO11 on 3D Copy-Paste synthetic data only
# for fraction in 10 20 50 100; do
#     python src/hyperparam_opt_YOLO.py --sweep_count 25 --fraction $fraction --data /home/data/configs/3d_CopyPaste_pace_v0-pace_eval.yaml --project pace-v2 --sweep_name 3D_CopyPaste-only-$fraction
# done

########### Sequential

# ## YOLO11 on Sequential: 2D CnP (50%) Extended -> PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name 2D_CNP_50-seq-pace_100 --model /home/wandb-runs/pace-v2-extended/2D_CNP-only-50/yejjmzte/fresh-sweep-13_extended/weights/best.pt

## YOLO11 on Sequential: Diffusion (100%) Extended -> PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name diffusion_100-seq-pace_100 --model /home/wandb-runs/pace-v2-extended/diffusion-only-100/sycnu5tm/comfy-sweep-20_extended/weights/best.pt

# ## YOLO11 on Sequential: 3D Render (100%) Extended -> PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name 3D_render_100-seq-pace_100 --model /home/wandb-runs/pace-v2-extended/3D_render-only-100/xi0kqli6/grateful-sweep-17_extended/weights/best.pt

# ## YOLO11 on Sequential: 3D Copy-Paste (100%) Extended -> PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name 3D_CopyPaste_100-seq-pace_100 --model /home/wandb-runs/pace-v2/3D_CopyPaste-only-100/kq2mo42m/misty-sweep-8/weights/best.pt

## YOLO11 on Sequential: 3D Copy-Paste (50%) Extended -> PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name 3D_CopyPaste_50-seq-pace_100 --model /home/wandb-runs/pace-v2-extended/3D_CopyPaste-only-50/ib5p0w19/sleek-sweep-10_extended/weights/best.pt

########### Mixed

# ## YOLO11 on 2D CnP (50%) mixed with PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/cnp_pace_v2_50-mix-pace_eval.yaml --project pace-v2 --sweep_name 2D_CNP-50-mixed

# ## YOLO11 on Diffusion (100%) mixed with PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/diffusion_v0-mix-pace_eval.yaml --project pace-v2 --sweep_name diffusion-mixed

# ## YOLO11 on 3d_render (100%) mixed with PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/3d_pace_v0-mix-pace_eval.yaml --project pace-v2 --sweep_name 3D_render-mixed

## YOLO11 on 3d_CopyPaste (100%) mixed with PACE (100%)
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/3d_CopyPaste_pace_v0-mix-pace_eval.yaml --project pace-v2 --sweep_name 3D_CP-mixed

## TESTING NEW AUTO-SEQUENTIAL FEATURE
# python src/hyperparam_opt_YOLO.py --sweep_count 25 --data /home/data/configs/pace_v2.yaml --project pace-v2 --sweep_name test-diffusion_100-seq-pace_100 --parent_sweep_name_dir /home/wandb-runs/pace-v2-extended/diffusion-only-100/

python src/post-sweep-testing.py 