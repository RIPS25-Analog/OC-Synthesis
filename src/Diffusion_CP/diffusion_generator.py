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
from pyblur import LinearMotionBlur

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


def randomAngle(kerneldim):
	"""Returns a random angle used to produce motion blurring

	Args:    kerneldim (int): size of the kernel used in motion blurring
	Returns: random angle (int)
	""" 
	kernelCenter = int(math.floor(kerneldim/2))
	numDistinctLines = kernelCenter * 4
	validLineAngles = np.linspace(0,180, numDistinctLines, endpoint=False)
	return int(np.random.choice(validLineAngles))


def LinearMotionBlur3C(img):
	"""Performs motion blur on an image with 3 channels. Used to simulate blurring caused due to motion of camera.

	Args: img(NumPy Array): Input image with 3 channels
	Returns: Image: Blurred image by applying a motion blur with random parameters
	"""
	lineLengths = [3,5,7,9]
	lineTypes = ["right", "left", "full"]
	lineLength = np.random.choice(lineLengths)
	lineType = np.random.choice(lineTypes)
	lineAngle = randomAngle(lineLength)
	blurred_img = img
	
	for i in range(3):
		blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
	
	blurred_img = Image.fromarray(blurred_img, 'RGB')
	return blurred_img


def get_mask_overlap(mask1, mask2):
	'''Check if two masks overlap or not. This is determined by maximum allowed IOU between masks.
	   If IOU is less than the max allowed IOU then masks don't overlap

	Args: mask1(NumPy Array) & mask2(NumPy Array): Mask 1 & 2
	Returns: True if masks overlap else False
	'''
	m1_sum = mask1.sum()
	m2_sum = mask2.sum()
	assert m1_sum > 0 and m2_sum > 0, "One of the masks is empty, cannot compute IOU"
	intersection = np.logical_and(mask1, mask2)
	iou1 = np.sum(intersection) / m1_sum
	iou2 = np.sum(intersection) / m2_sum
	return max(iou1, iou2)


def trim_img_n_mask(img_target, img_source, img_mask, offset):
	'''Creates a mask for the source image to be blended on the target image.
	   The mask is created by padding the source image mask with zeros according to the x, y offsets
		Also crops the source image to fit in the target image
	'''
	x, y = offset
	h_mask, w_mask = img_mask.shape
	h_target, w_target, _ = img_target.shape

	BOUNDARY_MARGIN = 2
	hd0 = max(BOUNDARY_MARGIN, -y)
	wd0 = max(BOUNDARY_MARGIN, -x)

	hd1 = h_mask - max(h_mask + y - h_target, 0) - BOUNDARY_MARGIN
	wd1 = w_mask - max(w_mask + x - w_target, 0) - BOUNDARY_MARGIN

	mask = np.zeros((h_mask, w_mask))
	mask[img_mask > 0] = 1

	mask = mask[hd0:hd1, wd0:wd1]
	src = img_source[hd0:hd1, wd0:wd1]

	# fix offset
	offset_adj = (max(x, 0), max(y, 0))

	# remove edge from the mask so that we don't have to check the
	# edge condition
	mask[:, -1] = 0
	mask[:, 0] = 0
	mask[-1, :] = 0
	mask[0, :] = 0

	return src, mask, offset_adj


def get_annotation_from_mask(mask, scale=1.0):
	'''Given a mask, this returns the bounding box annotations

	Args:
		mask(NumPy Array): Array with the mask
	Returns:
		tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
	'''
	rows = np.any(mask, axis=1)
	cols = np.any(mask, axis=0)
	assert len(np.where(rows)[0]) > 0, f"Found an all black mask file: {mask}"

	ymin, ymax = np.where(rows)[0][[0, -1]]
	xmin, xmax = np.where(cols)[0][[0, -1]]
	return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)


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


def PIL2array1C(img):
	'''Converts a PIL image to NumPy Array

	Args: img(PIL Image): Input PIL image
	Returns: NumPy Array: Converted image
	'''
	return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])


def PIL2array3C(img):
	'''Converts a PIL image to NumPy Array

	Args: img(PIL Image): Input PIL image
	Returns: NumPy Array: Converted image
	'''
	return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


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

    df_image_scale_by_ratio = NODE_CLASS_MAPPINGS["DF_Image_scale_by_ratio"]()
    df_image_scale_by_ratio_657 = df_image_scale_by_ratio.upscale(
        upscale_by=1,
        upscale_method="nearest-exact",
        crop="disabled",
        image=get_value_at_index(layerutility_colorimage_v2_655, 0),
    )

    easy_imagesize = NODE_CLASS_MAPPINGS["easy imageSize"]()
    easy_imagesize_1153 = easy_imagesize.image_width_height(
        image=get_value_at_index(df_image_scale_by_ratio_657, 0)
    )

    jwintegermin = NODE_CLASS_MAPPINGS["JWIntegerMin"]()
    jwintegermin_1366 = jwintegermin.execute(
        a=get_value_at_index(easy_imagesize_1153, 0),
        b=get_value_at_index(easy_imagesize_1153, 1),
    )

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
        resolution=get_value_at_index(jwintegermin_1366, 0),
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


def prearrange_objects(objects, img_file, anno_files, label_map, w, h, scale_augment, rotation_augment, blending_list, allow_full_occlusion):
    '''Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        img_file(str): Image file name
        anno_files(str): Annotation file names
        bg_file(str): Background image path 
        w(int): Width of synthesized image
        h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        dont_occlude_much(bool): Generate images with occlusion
    '''
    w = WIDTH
    h = HEIGHT
    blending_list = BLENDING_LIST

    if os.path.exists(anno_files[0]):
        logging.info(f"Annotation file {anno_files[0]} already exists, skipping this image")
        return anno_files[0]

    logging.info(f"Working on annotations {anno_files} which has objects: {objects}")

    assert len(objects) > 0

    logging.info('Creating a new image now...')
	
    for attempt in range(MAX_ATTEMPTS_TO_SYNTHESIZE):
        logging.info(f'\tStarting {attempt}th attempt to synthesize this image...')
        already_syn_objs = []
        all_objects_success = True
        objs_n_masks = [] # reset the list of objects and masks to try again

        # Try to place each object that's been assigned to this image (or skip if unplaceable)
        for idx, obj in enumerate(objects):
            logging.info(f'\tStarting object {obj[0]}...')
            foreground = Image.open(obj[0]).convert('RGB')
            foreground = ImageOps.exif_transpose(foreground)  # fix orientation if needed
            mask_file =  os.path.join(os.path.dirname(obj[0]), os.path.basename(obj[0]).split('.')[0] + '_mask.png').replace('/images', '/masks')

            original_mask = Image.open(mask_file)
            mask = original_mask
            mask_arr = PIL2array1C(original_mask)
            if INVERTED_MASK:
                mask = Image.fromarray(255-mask_arr).convert('1')

            dilated_foreground = foreground.copy()

            assert mask.size == foreground.size, f"Mask size {mask.size} does not match foreground size {foreground.size} for object {obj[0]}"
            
            o_w, o_h = foreground.size

            if rotation_augment:
                # logging.info(f'\t\tRotating object of size {o_w}x{o_h}...')
                rot_degrees = random.randint(-MAX_DEGREES, MAX_DEGREES)

                foreground = foreground.rotate(rot_degrees, expand=True)
                dilated_foreground = dilated_foreground.rotate(rot_degrees, expand=True)
                mask = mask.rotate(rot_degrees, expand=True)

                dilated_mask = Image.fromarray(cv2.dilate(mask_arr, np.ones((20,20), np.uint8), iterations=1), 'L')
                if INVERTED_MASK:
                    dilated_mask = Image.fromarray(PIL2array1C(dilated_mask)).convert('1')
                
                xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
                xmin_d, xmax_d, ymin_d, ymax_d = get_annotation_from_mask(dilated_mask)

                if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
                    raise ValueError(f"Invalid mask for object {obj[0]}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

                # logging.info(f'\t\tObject {obj[0]} mask annotation: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}')
                # logging.info(f'\t\tObject {obj[0]} dilated mask annotation: xmin={xmin_d}, ymin={ymin_d}, xmax={xmax_d}, ymax={ymax_d}')

                # logging.info(f'\t\t Foreground size before crop: {foreground.size} | Dilated foreground size before crop: {dilated_foreground.size}')
                # logging.info(f'\t\t Mask size before crop: {mask.size} | Dilated mask size before crop: {dilated_mask.size}')

                area_ratio = (xmax-xmin)*(ymax-ymin) / (o_w*o_h)
                foreground = foreground.crop((xmin, ymin, xmax, ymax))
                dilated_foreground = dilated_foreground.crop((xmin_d, ymin_d, xmax_d, ymax_d))
                mask = mask.crop((xmin, ymin, xmax, ymax))
                dilated_mask = dilated_mask.crop((xmin_d, ymin_d, xmax_d, ymax_d))

                # logging.info(f'\t\t Foreground size after crop: {foreground.size} | Dilated foreground size after crop: {dilated_foreground.size}')
                # logging.info(f'\t\t Mask size after crop: {mask.size} | Dilated mask size after crop: {dilated_mask.size}')

                o_w, o_h = foreground.size
                logging.info(f'\t\tObject rotated by {rot_degrees} degrees, new size: {o_w}x{o_h}')

            if o_w < 20 or o_h < 20:
                logging.warning(f'\t\tObject (after crop n rotate) is quite small!! ({o_w}x{o_h})...')

            if scale_augment:
                ACTUAL_MIN_SCALE = MIN_SCALED_DIM / min(o_w, o_h) # every object should be at least MIN_SCALED_DIM pixels in width/height
                ACTUAL_MAX_SCALE = min(w,h) / max(o_w, o_h) # every object should be at most min(w,h) pixels in width/height
                # the *0.9 ensures the object isn't exactly as wide/tall as the background since that may still give errors

                # scale the object so it occupies similar fraciton of the image as it did in the original foreground image
                length_scale = ((w*h) / (original_mask.size[0] * original_mask.size[1]))**0.5 
                scale = random.uniform(max(MIN_SCALE*length_scale, ACTUAL_MIN_SCALE), min(ACTUAL_MAX_SCALE, MAX_SCALE*length_scale))*0.9
                o_w, o_h = int(scale*o_w), int(scale*o_h)
                assert w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0, "Invalid object dimensions after scaling"

                foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
                mask = mask.resize((o_w, o_h), Image.LANCZOS)
                dilated_mask = dilated_mask.resize((o_w, o_h), Image.LANCZOS)
                
            # Compare current mask with all previous masks to avoid excess occlusion
            xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
            for placement_attempt in range(MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE):
                logging.info('\t\tStarting an object placement attempt...')
                x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
                y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))

                # if occlusion is allowed don't bother checking for overlap between objects
                if allow_full_occlusion:
                    break

                # Check if the object overlaps with any of the already synthesized objects
                obj_placement_is_valid = True

                trim_x_min = max(0, -x)
                trim_x_max = min(o_w, w - x)
                trim_y_min = max(0, -y)
                trim_y_max = min(o_h, h - y)

                trim_o_w = trim_x_max - trim_x_min
                trim_o_h = trim_y_max - trim_y_min

                paste_x_min = max(0, x)
                paste_x_max = paste_x_min + trim_o_w
                paste_y_min = max(0, y)
                paste_y_max = paste_y_min + trim_o_h

                assert paste_x_max <= w and paste_y_max <= h, f"Invalid paste coordinates: {paste_x_min}, {paste_y_min}, {paste_x_max}, {paste_y_max} for object {obj[0]} with size {o_w}x{o_h} at position ({x}, {y})"
                
                mask_array = np.zeros((h, w), dtype=np.uint8)
                trimmed_mask = PIL2array1C(mask)[trim_y_min:trim_y_max, trim_x_min:trim_x_max]
                mask_array[paste_y_min:paste_y_max, paste_x_min:paste_x_max] = trimmed_mask
                mask_array = np.where(mask_array > 2, True, False)

                for prev_obj in already_syn_objs:
                    overlap = get_mask_overlap(mask_array, prev_obj)
                    if overlap > MAX_OCCLUSION_IOU:
                        logging.info('\t\t\tOcclusion found, trying again...')
                        obj_placement_is_valid = False
                        break

                if obj_placement_is_valid:
                    if not allow_full_occlusion:
                        already_syn_objs.append(mask_array)
                    break
                
            else: # if we reach here, it means we could not place the object after MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE
                logging.warning(f'\t\tCould not place object {obj[0]} after {MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE} attempts')
                all_objects_success = False
                break
            
            objs_n_masks.append((obj[1], (x, y, xmin, ymin, xmax, ymax), foreground, dilated_foreground, mask, mask_array, dilated_mask))

        # If we could place all objects, then we're done trying object placements
        if all_objects_success:
            break        

    background = Image.new('RGB', (w, h), (128, 128, 128))  # create a gray background
    latent_mask = np.zeros((h, w), dtype=np.uint8)  # create an empty mask

    synth_images = []
    synth_masks = []
    for i in range(len(blending_list)):
        synth_images.append(background.copy())
        synth_masks.append(latent_mask.copy())

    logging.info('Blending objects now...')
    # Start pasting and blending objects
    for idx, obj_n_mask in enumerate(objs_n_masks):
        obj_class, (x, y, xmin, ymin, xmax, ymax), foreground, dilated_foreground, mask, mask_array, dilated_mask = obj_n_mask
        # Paste image on different background copies according to the different blending modes
        for i in range(len(blending_list)):
            if blending_list[i] == 'none' or blending_list[i] == 'motion':
                synth_images[i].paste(foreground, (x, y), mask)
                synth_masks[i] = cv2.bitwise_or(mask_array.astype(np.uint8), synth_masks[i])

            elif blending_list[i] == 'gaussian':
                blurred_mask = Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2))
                synth_images[i].paste(foreground, (x, y), blurred_mask)
                synth_masks[i] = cv2.bitwise_or(mask_array.astype(np.uint8), synth_masks[i])

            elif blending_list[i] == 'box':
                blurred_mask = Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3)))
                synth_images[i].paste(foreground, (x, y), blurred_mask)
                synth_masks[i] = cv2.bitwise_or(mask_array.astype(np.uint8), synth_masks[i])
        
        # Save annotations in text file
        class_num = [key for key in label_map if label_map[key] == obj_class][0]
        xmin = max(1, x+xmin)
        xmax = min(w, x+xmax)
        ymin = max(1, y+ymin)
        ymax = min(h, y+ymax)
        string = f"{class_num} {(xmin+xmax)/(2*w)} {(ymin+ymax)/(2*h)} {(xmax-xmin)/w} {(ymax-ymin)/h}\n"
        for anno_file in anno_files:
            if not os.path.exists(anno_file):
                os.system(f"touch {anno_file}")  # create the file if it doesn't exist
            logging.info(f'\t\tWriting annotation for object {obj_class} at {x}, {y} with mask {xmin, ymin, xmax, ymax} to {anno_file} if it is in {SELECTED_CLASSES}')
            if obj_class in SELECTED_CLASSES:
                with open(anno_file, "a") as f:
                    f.write(string)
    return synth_images, synth_masks


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
