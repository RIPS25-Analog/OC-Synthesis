# Synthetic Data Generation with Diffusion 

This code is used to generate synthetic images for training object detection models on a set of target classes. Given images of objects and masks isolating the objects from their backgrounds, it creates diverse arrangements of the objects and uses an image generation model to synthesize a contextually coherent background for the foreground objects. The user supplies reference images (e.g. other training images not containing the target classes) which control the style in which the backgrounds are generated. The output is a directory containing synthesized image and label directories, where annotations are given in YOLO format.

## Foreground Object and Background Data
This method requires a folder with images of foreground objects and a mask for each image, as well as a folder with reference background images:
```
└── /home/data/raw/[dataset_name]
    ├── foreground_objects/
    |   ├── class_name_1
    |   |   ├── images              # images of the same file type (e.g: 4_5.png, 15_10.png)
    |   |   └── masks               # masks with same filenames as corresponding images plus _mask attached at end (e.g: 4_5_mask.png, 15_10_mask.png)
    |   ├── ...
    |   └── class_name_8
    └── backgrounds/                # backgrounds of the same file type (e.g: 1.png, 2.png)
```

## Installation 

Navigate to ``src/Diffusion-Copy-Paste`` in this repository and install the requirements:
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install onnxruntime-gpu
```

Next, clone the ComfyUI repository into your home folder (https://github.com/comfyanonymous/ComfyUI):

```
git clone https://github.com/comfyanonymous/ComfyUI.git
```

Using the Hugging Face CLI (https://huggingface.co/docs/huggingface_hub/en/guides/cli) or otherwise, download the following models into the indicated folders (within ``~/ComfyUI``):

Controlnet (save as ``models/controlnet/diffusion_pytorch_model_promax.safetensors``):
```
hf download xinsir/controlnet-union-sdxl-1.0 diffusion_pytorch_model_promax.safetensors --local-dir models/controlnet
```

Diffusion model (save as ``models/checkpoints/RealVisXL_V5.0_Lightning_fp16.safetensors``):
```
hf download SG161222/RealVisXL_V5.0_Lightning RealVisXL_V5.0_Lightning_fp16.safetensors --local-dir models/checkpoints
```

CLIP-vision (save as ``models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors``)
```
hf download h94/IP-Adapter models/image_encoder/model.safetensors --local-dir models/clip_vision
mv models/clip_vision/models/image_encoder/model.safetensors models/clip_vision
mv models/clip_vision/model.safetensors models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
rm -rf models/clip_vision/models/
```

IP-Adapter (save as ``models/ipadapter/ip-adapter_sdxl_vit-h.bin``):
```
mkdir models/ipadapter
hf download h94/IP-Adapter sdxl_models/ip-adapter_sdxl_vit-h.bin --local-dir models/ipadapter
mv models/ipadapter/sdxl_models/ip-adapter_sdxl_vit-h.bin models/ipadapter/
rm -rf models/ipadapter/sdxl_models
```

LaMa Remover (save as ``custom_nodes/comfyui_lama_remover/ckpts/big-lama.pt``):
```
wget -P custom_nodes/comfyui_lama_remover/ckpts https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
```

Finally, install the custom nodes by running the following commands in the ComfyUI directory.
```
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git custom_nodes/comfyui_ipadapter_plus

git clone https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/comfyui_essentials

git clone https://github.com/chflame163/ComfyUI_LayerStyle.git custom_nodes/comfyui_layerstyle

git clone https://github.com/yolain/ComfyUI-Easy-Use.git custom_nodes/comfyui-easy-use

git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git custom_nodes/comfyui_controlnet_aux

git clone https://github.com/Layer-norm/comfyui-lama-remover.git custom_nodes/comfyui_lama_remover
```


## Setting up defaults
The defaults.py file allows selection of a set of target classes that will be annotated in the final output (all other classes will be treated as distractors). It also contains different image generation parameters that can be varied to produce scenes with different target object sizes, clutter/occlusion levels, etc.

## Running the Script
```
usage: diffusion_generator.py [-h]
                              --root ROOT 
                              --background_images BACKGROUND_IMAGES
                              --output_dir OUTPUT_DIR 
                              [--pos_prompts POS_PROMPTS] [--neg_prompts NEG_PROMPTS] [--seed SEED] [--steps STEPS] 
                              [--max_obj_images MAX_OBJ_IMAGES] [--num_foregrounds NUM_FOREGROUNDS] [--num_backgrounds NUM_BACKGROUNDS] [--num_seeds NUM_SEEDS] 
                              [--selected] [--no_scale] [--no_rotation] [--allow_full_occlusion] [--no_distractors] [--dont_parallelize]

Create dataset with diffusion-generated backgrounds

options:
  -h, --help            show this help message and exit
  --root ROOT           Path to the input root directory containing images and masks (default: None)
  --background_images BACKGROUND_IMAGES
                        Path to the background image directory (default: None)
  --output_dir OUTPUT_DIR
                        Directory to save the output images (default: None)
  --pos_prompts POS_PROMPTS
                        Positive prompts for the model (default: ['near view'])
  --neg_prompts NEG_PROMPTS
                        Negative prompts for the model (default: ['blurry, low quality, bad resolution'])
  --seed SEED           Random seed for reproducibility (default: 472036261688012)
  --steps STEPS         Number of diffusion steps (default: 6)
  --max_obj_images MAX_OBJ_IMAGES
                        Maximum number of object images to use overall (default: 1000000000)
  --num_foregrounds NUM_FOREGROUNDS
                        Number of foregrounds to generate (default: 1000.0)
  --num_backgrounds NUM_BACKGROUNDS
                        Number of backgrounds to use for diffusion prompting (default: 1000)
  --num_seeds NUM_SEEDS
                        Number of times to change the random seed for generating a background with diffusion (default: 10)
  --selected            Keep only selected instances in the test dataset. Default is to keep all instances in the root directory (default: False)
  --no_scale            Remove scale augmentation. Default is to add scale augmentation. (default: False)
  --no_rotation         Remove rotation augmentation. Default is to add rotation data augmentation. (default: False)
  --allow_full_occlusion
                        Allow complete occlusion between objects (faster). Default is to avoid high occlusions (as defined by MAX_OCCLUSION_IOU) (default: False)
  --no_distractors      Don't add distractors objects. Default is to add distractors (default: False)
  --dont_parallelize    Run the dataset generation in serial mode. Default is to run in parallel mode (default: False)
```

## Training an object detector
The code produces all the files required to train an object detection model, such as YOLO. The following is the folder structure of the output images and annotations
```
└── /home/data/[output_dir]/
    ├── images/         # images of the same file type (e.g: 4_5.png, 15_10.png)
    └── labels/         # labels in darknet format (class_id, x, y, w, h) with same filenames as corresponding images (e.g: 4_5.txt, 15_10.txt)
```

## References
This approach adapts some of the code from a paper of Dwibedi et al. for object arrangement:
[Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection](https://vision.unipv.it/CV/materiale2017-18/fourthchoice/Dwibedi_Cut_Paste_and_ICCV_2017_paper.pdf).

We also access certain core and custom nodes from the ComfyUI backbone for running the image generation model and certain image processing tasks:
[ComfyUI Docs](https://docs.comfy.org/).

In particular, we make use of the following models and adapters:
[RealVisXL v5.0 Lighting](https://civitai.com/models/139562/realvisxl-v50),
[ControlNet](https://github.com/lllyasviel/ControlNet),
[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter),
[LaMa Remover](https://github.com/Layer-norm/comfyui-lama-remover).
