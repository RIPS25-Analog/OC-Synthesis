import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import cv2
import numpy as np
import argparse

from matplotlib import pyplot as plt
import yaml
from collections import namedtuple

import glob
import os
from multiprocessing import Pool
from functools import partial
import signal
import time

import math
import numpy as np
import random
from PIL import Image, ImageOps, ImageChops
import cv2
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import GaussianBlur

from defaults import *
from cut_and_paste import *

seed = 2
np.random.seed(seed)
random.seed(seed)

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

assert os.path.exists(os.path.join(os.path.expanduser("~"), "ComfyUI")), "Cannot find ComfyUI directory"
sys.path.append(os.path.join(os.path.expanduser("~"), "ComfyUI"))


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    # try:
    #     from main import load_extra_path_config
    # except ImportError:
    #     print(
    #         "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
    #     )
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()
from nodes import NODE_CLASS_MAPPINGS


def write_yaml_file(exp_dir, labels, label_map):
	'''Writes the .yaml for YOLO training.

	Args:
		exp_dir(string): Experiment directory where all the generated images, annotation and imageset
						 files will be stored
		labels(list): List of labels. This will be useful while training an object detector
	'''
	unique_labels = sorted(set(labels))
	yaml_filename = f"{'_'.join(exp_dir.split('/')[-2:])}.yaml"  # join raw_data and processed_data name
	yaml_path = os.path.join('/home/data/configs', yaml_filename)
	
	data = {
		'path': str(exp_dir),
		'train': os.path.join(exp_dir, 'train'),
		'val': os.path.join(exp_dir, 'val'),
		'test': os.path.join(exp_dir, 'test'),
		'nc': len(unique_labels),
		'names': {i: label for i, label in enumerate(unique_labels) if label in SELECTED_CLASSES},
	}

	with open(yaml_path, 'w') as f:
		yaml.dump(data, f, sort_keys=False)


def init_worker():
	''' Catch Ctrl+C signal to terminate workers '''
	signal.signal(signal.SIGINT, signal.SIG_IGN)


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes - this is an async function so we need to run it in the loop
    loop.run_until_complete(init_extra_nodes())


def load_checkpoints_n_models(background_image):
    """Load checkpoints from the checkpoints folder and add them to NODE_CLASS_MAPPINGS"""
    controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
    controlnetloader_688 = controlnetloader.load_controlnet(
        control_net_name="diffusion_pytorch_model_promax.safetensors"
    )

    checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    checkpointloadersimple_714 = checkpointloadersimple.load_checkpoint(
        ckpt_name="RealVisXL_V5.0_Lightning_fp16.safetensors"
    )

    differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
    differentialdiffusion_1043 = differentialdiffusion.apply(
        model=get_value_at_index(checkpointloadersimple_714, 0)
    )

    ipadapterloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
    ipadapter_loaded = ipadapterloader.load_models(
        get_value_at_index(differentialdiffusion_1043, 0), 
        "STANDARD (medium strength)", 
        provider="CUDA"
    )

    ipadapter = NODE_CLASS_MAPPINGS["IPAdapter"]()
    ipadapter_applied = ipadapter.apply_ipadapter(
        weight=2,
        start_at=0,
        end_at=1,
        weight_type="standard",
        model=get_value_at_index(ipadapter_loaded, 0),
        ipadapter=get_value_at_index(ipadapter_loaded, 1),
        image=background_image
    )

    return (
        get_value_at_index(controlnetloader_688, 0),
        get_value_at_index(ipadapter_applied, 0),
        get_value_at_index(checkpointloadersimple_714, 1),
        get_value_at_index(checkpointloadersimple_714, 2)
    )


def clip_text_encode(pos_prompt, neg_prompt, clip):
    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    cliptextencode_pos = cliptextencode.encode(
        text=pos_prompt,
        clip=clip,
    )

    cliptextencode_neg = cliptextencode.encode(
        text=neg_prompt, clip=clip
    )

    return get_value_at_index(cliptextencode_pos, 0), get_value_at_index(cliptextencode_neg, 0)


def preprocess_images_masks(loaded_image, loaded_mask):

    sdxlemptylatentsizepicker = NODE_CLASS_MAPPINGS["SDXLEmptyLatentSizePicker+"]()
    sdxlemptylatentsizepicker_616 = sdxlemptylatentsizepicker.execute(
        resolution="1024x1024",
        batch_size=1,
        width_override=WIDTH,
        height_override=HEIGHT,
    )

    layerutility_colorimage_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ColorImage V2"]()
    layerutility_colorimage_v2_655 = layerutility_colorimage_v2.color_image_v2(
        size="custom",
        custom_width=get_value_at_index(sdxlemptylatentsizepicker_616, 1),
        custom_height=get_value_at_index(sdxlemptylatentsizepicker_616, 2),
        color="#7f7f7f",
    )

    easy_imagesize = NODE_CLASS_MAPPINGS["easy imageSize"]()
    easy_imagesize_1153 = easy_imagesize.image_width_height(
        image=get_value_at_index(layerutility_colorimage_v2_655, 0)
    )

    resolution = min(get_value_at_index(easy_imagesize_1153, 0), get_value_at_index(easy_imagesize_1153, 1))

    # background_tensor = loaded_image
    # background_image = background_tensor.cpu().numpy()
    # background_image = np.clip(background_image, 0, 1) * 255
    # background_image = background_image.astype(np.uint8)
    # background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("test_image.png", background_image)

    # background_tensor = loaded_mask
    # background_image = background_tensor.cpu().numpy()
    # background_image = np.clip(background_image, 0, 1) * 255
    # background_image = background_image.astype(np.uint8)
    # background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("test_mask.png", background_image)

    cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()
    cannyedgepreprocessor_800 = cannyedgepreprocessor.execute(
        low_threshold=50,
        high_threshold=100,
        resolution=resolution,
        image=loaded_image,
    )

    emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    latent_image = emptylatentimage.generate(
        width=get_value_at_index(easy_imagesize_1153, 0),
        height=get_value_at_index(easy_imagesize_1153, 1),
        batch_size=1,
    )

    return (
        get_value_at_index(latent_image, 0),
        loaded_image,
        loaded_mask,
        get_value_at_index(cannyedgepreprocessor_800, 0)
    )


def generate_background(model, controlnet, clip_pos, clip_neg, edges, latent_image, vae, seed=472036261688017, steps=6, cfg=2.0):
    controlnetapplier = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
    controlnet_applied = controlnetapplier.apply_controlnet(clip_pos, clip_neg, controlnet, edges, 1.0, 0.0, 1.0, vae=vae)

    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    params = {
        "model": model,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "dpmpp_sde",
        "scheduler": "karras",
        "positive": get_value_at_index(controlnet_applied, 0),
        "negative": get_value_at_index(controlnet_applied, 1),
        "latent_image": latent_image,
        "denoise": 1.0
    }
    ksampled = ksampler.sample(
        params["model"],
        seed=params["seed"],
        steps=params["steps"],
        cfg=params["cfg"],
        sampler_name=params["sampler_name"],
        scheduler=params["scheduler"],
        positive=params["positive"],
        negative=params["negative"],
        latent_image=params["latent_image"],
        denoise=params["denoise"]
    )
    
    VAE_decoder = NODE_CLASS_MAPPINGS["VAEDecode"]()
    generated_background = VAE_decoder.decode(vae, get_value_at_index(ksampled, 0))[0]
    return get_value_at_index(generated_background, 0)


def inpaint_image(model, controlnet, clip_pos, clip_neg, image, foreground, mask, vae, seed=472036261687966, steps=4, cfg=2.0):
    controlnetapplier = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
    controlnet_applied = controlnetapplier.apply_controlnet(clip_pos, clip_neg, controlnet, foreground, 1.0, 0.0, 1.0, vae=vae)

    inpaintmodelconditioner = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    inpainted_conditioning = inpaintmodelconditioner.encode(
        get_value_at_index(controlnet_applied, 0),
        get_value_at_index(controlnet_applied, 1),
        image,
        vae,
        mask
    )

    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    params = {
        "model": model,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "dpmpp_sde",
        "scheduler": "karras",
        "positive": get_value_at_index(inpainted_conditioning, 0),
        "negative": get_value_at_index(inpainted_conditioning, 1),
        "latent_image": get_value_at_index(inpainted_conditioning, 2),
        "denoise": 0.8
    }
    ksampled = ksampler.sample(
        params["model"],
        seed=params["seed"],
        steps=params["steps"],
        cfg=params["cfg"],
        sampler_name=params["sampler_name"],
        scheduler=params["scheduler"],
        positive=params["positive"],
        negative=params["negative"],
        latent_image=params["latent_image"],
        denoise=params["denoise"]
    )
    
    VAE_decoder = NODE_CLASS_MAPPINGS["VAEDecode"]()
    generated_inpainted_image = VAE_decoder.decode(vae, get_value_at_index(ksampled, 0))[0]
    return generated_inpainted_image


def remove_and_paste(generated_background, mask, image, model, controlnet, cliptextencode_pos, cliptextencode_neg, vae):
    """Remove the generated foreground and paste the original foreground."""
    layerutility_colorimage_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ColorImage V2"]()
    layerutility_colorimage_v2_2 = layerutility_colorimage_v2.color_image_v2(
        size="custom", custom_width=1024, custom_height=1024, color="#7f7f7f"
    )

    imagedetailtransfer = NODE_CLASS_MAPPINGS["easy imageDetailTransfer"]()
    imagedetailtransfer_2 = imagedetailtransfer.transfer(
        target=generated_background.unsqueeze(0),
        source=torch.permute(image, (1, 2, 0)).unsqueeze(0) / 255,
        mode="add",
        blur_sigma=40.00,
        blend_factor=1.000,
        save_prefix=None,
        image_output="Preview",
        mask=mask.unsqueeze(0).to(torch.float32)
    )
    
    # img_2 = get_value_at_index(imagedetailtransfer_2, 0)
    # img_2 = img_2.cpu().numpy()[0]
    # img_2 = np.clip(img_2, 0, 1) * 255
    # img_2 = img_2.astype(np.uint8)
    # img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("imagedetailtransfer_2.png", img_2)

    imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
    imagecompositemasked_4 = imagecompositemasked.composite(
        x=0,
        y=0,
        resize_source=False,
        destination=get_value_at_index(layerutility_colorimage_v2_2, 0),
        source=get_value_at_index(imagedetailtransfer_2, 0),
    )

    # img_3 = get_value_at_index(imagecompositemasked_4, 0)
    # img_3 = img_3.cpu().numpy()[0]
    # img_3 = np.clip(img_3, 0, 1) * 255
    # img_3 = img_3.astype(np.uint8)
    # img_3 = cv2.cvtColor(img_3, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("imagecompositemasked_3.png", img_3)

    lamaremover = NODE_CLASS_MAPPINGS["LamaRemover"]()
    removed_fg = lamaremover.lama_remover(
        [get_value_at_index(imagecompositemasked_4, 0).squeeze(0)],
        [255*(1-mask)],
        mask_threshold=250,
        gaussblur_radius=7,
        invert_mask=True,
    )

    # removed = get_value_at_index(removed_fg, 0)
    # removed = removed.cpu().numpy()[0]
    # removed = np.clip(removed, 0, 1) * 255
    # removed = removed.astype(np.uint8)
    # removed = cv2.cvtColor(removed, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("removed_foreground.png", removed)  

    imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
    composited_image_1 = imagecompositemasked.composite(
        get_value_at_index(removed_fg, 0),
        generated_background.unsqueeze(0),
        0,
        0,
        False,
        mask=mask.unsqueeze(0)
    )

    mask = GaussianBlur(kernel_size=(5, 5), sigma=(2.0, 2.0))(mask.to(torch.float32).unsqueeze(0)).squeeze(0)
    image = GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0))(image.unsqueeze(0)).squeeze(0)

    imagedetailtransfer = NODE_CLASS_MAPPINGS["easy imageDetailTransfer"]()
    detail_transfer_params = {
        "target": get_value_at_index(composited_image_1, 0),
        "source": torch.permute(image, (1, 2, 0)).unsqueeze(0) / 255,
        "mode": "soft_light",
        "blur_sigma": 5.00,
        "blend_factor": 1.000,
        "image_output": "Preview",
        "save_prefix": None,
        "mask": mask.unsqueeze(0).to(torch.float32)
    }
    detail_transferred_1 = imagedetailtransfer.transfer(
        detail_transfer_params["target"],
        detail_transfer_params["source"],
        detail_transfer_params["mode"],
        detail_transfer_params["blur_sigma"],
        detail_transfer_params["blend_factor"],
        detail_transfer_params["image_output"],
        detail_transfer_params["save_prefix"],
        mask=detail_transfer_params["mask"]
    )

    # img_4 = get_value_at_index(detail_transferred_1, 0)
    # img_4 = img_4.cpu().numpy()[0]
    # img_4 = np.clip(img_4, 0, 1) * 255
    # img_4 = img_4.astype(np.uint8)
    # img_4 = cv2.cvtColor(img_4, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("imagecompositemasked_4.png", img_4)

    detail_transfer_params = {
        "target": get_value_at_index(detail_transferred_1, 0),
        "source": torch.permute(image, (1, 2, 0)).unsqueeze(0) / 255,
        "mode": "add",
        "blur_sigma": 40.00,
        "blend_factor": 1.000,
        "image_output": "Preview",
        "save_prefix": None,
        "mask": mask.unsqueeze(0).to(torch.float32)
    }
    detail_transferred_2 = imagedetailtransfer.transfer(
        detail_transfer_params["target"],
        detail_transfer_params["source"],
        detail_transfer_params["mode"],
        detail_transfer_params["blur_sigma"],
        detail_transfer_params["blend_factor"],
        detail_transfer_params["image_output"],
        detail_transfer_params["save_prefix"],
        mask=detail_transfer_params["mask"]
    )

    # img_5 = get_value_at_index(detail_transferred_2, 0)
    # img_5 = img_5.cpu().numpy()[0]
    # img_5 = np.clip(img_5, 0, 1) * 255
    # img_5 = img_5.astype(np.uint8)
    # img_5 = cv2.cvtColor(img_5, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("imagecompositemasked_5.png", img_5)
    
    # final_img = GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0))(get_value_at_index(detail_transferred_2, 0))
    # final_img = final_img.cpu().numpy()[0]
    # final_img = np.clip(final_img, 0, 1) * 255
    # final_img = final_img.astype(np.uint8)
    # final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("final_image_after_blur.png", final_img)

    return get_value_at_index(detail_transferred_2, 0)


def prearrange_objects_wrapper(params, w, h, scale_augment, rotation_augment, blending_list, allow_full_occlusion):
	''' Wrapper used to pass params to workers '''
	try:
		return prearrange_objects(*params, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment,
					 blending_list=blending_list, allow_full_occlusion=allow_full_occlusion)
	except Exception as e:
		logging.exception(f"Error while creating image annotation: {e}")
		if not args.parallelize:
			exit()


def generate_data(args):
    background_images = args.background_images
    output_dir = args.output_dir
    pos_prompts = args.pos_prompts
    neg_prompts = args.neg_prompts
    seed = args.seed
    steps = args.steps

    img_files = glob.glob(os.path.join(args.root, '*', 'images', '*'))[:args.max_obj_images]
    random.shuffle(img_files)
    class_names = [img_file.split('/')[-3] for img_file in img_files]

    unselected_classes = set(class_names) - set(SELECTED_CLASSES)
    all_classes = SELECTED_CLASSES + list(unselected_classes)
    label_map = {i: label for i, label in enumerate(all_classes)}

    write_yaml_file(output_dir, SELECTED_CLASSES, label_map)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    anno_dir = os.path.join(output_dir, 'labels')
    img_dir = os.path.join(output_dir, 'images')

    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)

    image_class_pairs = list(zip(img_files, class_names))
    random.shuffle(image_class_pairs)

    objects = list(zip(img_files, class_names))
    random.shuffle(objects)
    target_objects = [pair for pair in objects if pair[1] in SELECTED_CLASSES]
    distractor_objects = [pair for pair in objects if pair[1] not in SELECTED_CLASSES]

    assert len(target_objects) > 0, "No target objects found in the dataset"
    if args.add_distractors:
        assert len(distractor_objects) > 0, "No distractor objects found in the dataset"

    img_files = []
    params_list = []
    images = []
    masks = []
    for bg_idx in range(args.num_backgrounds):
        for img_idx in range(args.num_foregrounds):
            # Get list of objects
            selected_objects = []
            n_objects = random.randint(MIN_N_OBJECTS, MAX_N_OBJECTS)
            n_target_objects = random.randint(MIN_N_TARGET_OBJECTS, MAX_N_TARGET_OBJECTS)

            selected_objects.extend(random.sample(target_objects, n_target_objects))

            if args.add_distractors:
                selected_objects.extend(random.sample(distractor_objects, n_objects-n_target_objects))

            logging.info(f"Chosen objects: {selected_objects}")

            img_file = os.path.join(img_dir, f'bg_{bg_idx}_fg_{img_idx}_seed.jpg')
            anno_list = [os.path.join(anno_dir, f'bg_{bg_idx}_fg_{img_idx}_seed_{seed_idx}.txt') for seed_idx in range(args.num_seeds)]
            params_list.append((selected_objects, img_file, anno_list, label_map))

    if args.parallelize:
        logging.info(f"Parallelizing with {NUMBER_OF_WORKERS} workers")
        partial_func = partial(prearrange_objects_wrapper, w=WIDTH, h=HEIGHT, scale_augment=args.scale_augment, rotation_augment=args.rotation_augment, 
                               blending_list=BLENDING_LIST, allow_full_occlusion=args.allow_full_occlusion) 
        p = Pool(NUMBER_OF_WORKERS, init_worker)
        try:
            results = p.map(partial_func, params_list)
            images, masks = [imgs[0] for imgs, _ in results], [masks[0] for _, masks in results]
        except KeyboardInterrupt:
            logging.warning("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    else:
        logging.info("Not parallelizing, running in serial mode")
        for params in params_list:
            images_generated, masks_generated = prearrange_objects_wrapper(params, w=WIDTH, h=HEIGHT, scale_augment=args.scale_augment, rotation_augment=args.rotation_augment,
                                blending_list=BLENDING_LIST, allow_full_occlusion=args.allow_full_occlusion)
            images += images_generated
            masks += masks_generated

    seeds = [seed + i for i in range(args.num_seeds)]
    import_custom_nodes()
    with torch.inference_mode():
        assert len(images) == len(masks)
        for bg_idx in range(args.num_backgrounds):
            t0 = time.time()

            background_image = os.path.join(background_images, np.random.choice(os.listdir(background_images)))
            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            background_image = get_value_at_index(loadimage.load_image(image=background_image), 0)
            cv2.imwrite("test_background.png", cv2.cvtColor((background_image.squeeze(0).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            pos_prompt = np.random.choice(pos_prompts)
            neg_prompt = np.random.choice(neg_prompts)
            controlnet, realvisxl, clip, vae = load_checkpoints_n_models(background_image)

            t1 = time.time()
            logging.info(f"Time taken to load checkpoints and models: {t1 - t0:.2f} seconds")

            cliptextencode_pos, cliptextencode_neg = clip_text_encode(pos_prompt, neg_prompt, clip)

            t2 = time.time()
            logging.info(f"Time taken to encode prompts: {t2 - t1:.2f} seconds")

            images_slice = images[bg_idx * args.num_foregrounds:(bg_idx + 1) * args.num_foregrounds]
            masks_slice = masks[bg_idx * args.num_foregrounds:(bg_idx + 1) * args.num_foregrounds]
            for idx, foreground, foreground_mask in zip(range(args.num_foregrounds), images_slice, masks_slice):
                # foreground.save("test_foreground.png")
                # cv2.imwrite("test_foreground_mask.png", cv2.cvtColor(foreground_mask*255, cv2.COLOR_GRAY2BGR))
                for seed_idx, seed in enumerate(seeds):
                    t3 = time.time()
                    latent_image, image, mask, edges = preprocess_images_masks(pil_to_tensor(foreground), torch.tensor(foreground_mask))
                    t4 = time.time()
                    logging.info(f"Time taken to preprocess images and masks: {t4 - t3:.2f} seconds")

                    # mask_tensor = mask
                    # mask_tensor = mask_tensor.cpu().numpy()
                    # mask_tensor = np.clip(mask_tensor, 0, 1) * 255
                    # mask_tensor = mask_tensor.astype(np.uint8)
                    # mask_tensor = cv2.cvtColor(mask_tensor, cv2.COLOR_GRAY2BGR)
                    # cv2.imwrite("test_mask.png", mask_tensor)

                    # image_tensor = torch.permute(image, (1, 2, 0))
                    # image_tensor = image_tensor.cpu().numpy()
                    # # pil_to_tensor gives values in range [0, 255], not [0, 1]
                    # image_tensor = np.clip(image_tensor, 0, 255).astype(np.uint8)
                    # # Convert RGB to BGR for cv2.imwrite
                    # image_tensor_bgr = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite("test_image.png", image_tensor_bgr)

                    generated_background = generate_background(realvisxl, controlnet, cliptextencode_pos, cliptextencode_neg, edges, latent_image, vae, seed=seed, steps=steps)

                    t5 = time.time()
                    logging.info(f"Time taken to generate background: {t5 - t4:.2f} seconds")

                    # background_tensor = generated_background
                    # background_image = background_tensor.cpu().numpy()
                    # background_image = np.clip(background_image, 0, 1) * 255
                    # background_image = background_image.astype(np.uint8)
                    # background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite("generated_background.png", background_image)

                    final_image = remove_and_paste(generated_background, mask, image, realvisxl, controlnet, cliptextencode_pos, cliptextencode_neg, vae)

                    t6 = time.time()
                    logging.info(f"Time taken to inpaint and paste foreground: {t6 - t5:.2f} seconds")

                    final_image = final_image.cpu().numpy().squeeze(0)
                    assert np.max(final_image) <= 1.0 and np.min(final_image) >= 0.0, "Final image values are not in the range [0, 1]"
                    final_image = np.clip(final_image, 0, 1) * 255
                    final_image = final_image.astype(np.uint8)
                    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, 'images', f"bg_{bg_idx}_fg_{idx}_seed_{seed_idx}.png"), final_image)

                    t7 = time.time()
                    logging.info(f"Time taken in total: {t7 - t1:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset with diffusion-generated backgrounds",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=str, required=True, help="Path to the input root directory containing images and masks")
    parser.add_argument("--background_images", type=str, required=True, help="Path to the background image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images")
    parser.add_argument("--pos_prompts", type=str, default=["near view"], help="Positive prompts for the model")
    parser.add_argument("--neg_prompts", type=str, default=["blurry, low quality, bad resolution"], help="Negative prompts for the model")
    parser.add_argument("--seed", type=int, default=472036261688012, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=6, help="Number of diffusion steps")
    parser.add_argument("--max_obj_images",
      help="Maximum number of object images to use overall", default=int(1e9), type=int)
    parser.add_argument("--num_foregrounds",
        help="Number of foregrounds to generate", default=1e3, type=int)
    parser.add_argument("--num_backgrounds",
      help="Number of backgrounds to use for diffusion prompting", default=1000, type=int)
    parser.add_argument("--num_seeds",
        help="Number of times to change the random seed for generating a background with diffusion", default=10, type=int)
    parser.add_argument("--selected",
        help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory", action="store_true")
    parser.add_argument("--no_scale",
        help="Remove scale augmentation. Default is to add scale augmentation.", action="store_true")
    parser.add_argument("--no_rotation",
        help="Remove rotation augmentation. Default is to add rotation data augmentation.", action="store_true")
    parser.add_argument("--allow_full_occlusion",
        help="Allow complete occlusion between objects (faster). Default is to avoid high occlusions (as defined by MAX_OCCLUSION_IOU)", action="store_true")
    parser.add_argument("--no_distractors",
	  help="Don't add distractors objects. Default is to add distractors", action="store_true")
    parser.add_argument("--dont_parallelize",
	  help="Run the dataset generation in serial mode. Default is to run in parallel mode", action="store_true")
    args = parser.parse_args()
    kwargs = vars(args)

    args.parallelize = not args.dont_parallelize
    args.scale_augment = not args.no_scale
    args.rotation_augment = not args.no_rotation
    args.add_distractors = not args.no_distractors
    del args.no_scale
    del args.no_rotation
    del args.dont_parallelize
    del args.no_distractors

    import logging
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    log_file_path = os.path.join(args.output_dir, "diffusion_generator.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        filename=log_file_path, 
        encoding='utf-8', 
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w',  # overwrite mode for testing
        force=True  # force reconfiguration
    )
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info("Starting data generation with parameters:")
    for key, value in kwargs.items():
        logging.info(f"{key}: {value}")
    
    generate_data(args)
    logging.info("Data generation completed.")
