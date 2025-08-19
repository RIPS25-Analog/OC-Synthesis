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
import logging

import math
import numpy as np
import random
from PIL import Image, ImageOps
import cv2

from defaults import *
from pyblur_master.pyblur import LinearMotionBlur
from datetime import datetime

import post_processing

seed = 0
np.random.seed(seed)
random.seed(seed)

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

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
	
	blurred_img = Image.fromarray(blurred_img)#, 'RGB')
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

def write_yaml_file(exp_dir, label_map):
	'''Writes the .yaml for YOLO training.

	Args:
		exp_dir(string): Experiment directory where all the generated images, annotation and imageset
						 files will be stored
		labels(list): List of labels. This will be useful while training an object detector
	'''
	yaml_filename = f"{'_'.join(exp_dir.split('/')[-2:])}.yaml" # join raw_data and processed_data name
	yaml_path = os.path.join('/home/data/configs', yaml_filename)
	
	data = {
		'train': os.path.join(exp_dir, 'train'),
		'val': os.path.join(exp_dir, 'val'),
		'test': os.path.join(exp_dir, 'test'),
		'nc': len(SELECTED_CLASSES),
		'names': {label_id: class_name for label_id, class_name in label_map.items() if class_name in SELECTED_CLASSES},
	}

	with open(yaml_path, 'w') as f:
		yaml.dump(data, f, sort_keys=False)

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

def create_image_anno_wrapper(params, w, h, scale_augment, rotation_augment, blending_list, allow_full_occlusion):
	''' Wrapper used to pass params to workers '''
	try:
		create_image_anno(*params, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment,
					 blending_list=blending_list, allow_full_occlusion=allow_full_occlusion)
	except Exception as e:
		logging.exception(f"Error while creating image annotation: {e}")
		if not args.parallelize:
			exit()

def create_image_anno(objects, img_file, anno_file, bg_file, label_map, w, h, scale_augment, rotation_augment, blending_list, allow_full_occlusion):
	'''Add data augmentation, synthesizes images and generates annotations according to given parameters

	Args:
		objects(list): List of objects whose annotations are also important
		img_file(str): Image file name
		anno_file(str): Annotation file name
		bg_file(str): Background image path 
		w(int): Width of synthesized image
		h(int): Height of synthesized image
		scale_augment(bool): Add scale data augmentation
		rotation_augment(bool): Add rotation data augmentation
		blending_list(list): List of blending modes to synthesize for each image
		allow_full_occlusion(bool): Generate images without checking for too much occlusion
	'''
	assert 'none' in img_file, "Base image file should contain 'none' blending mode in its name"
	assert len(objects) > 0, "No objects provided for synthesis, cannot create image"

	if os.path.exists(anno_file):
		logging.info(f"Annotation file {anno_file} already exists, skipping this image")
		return anno_file

	logging.info(f"Working on annotation {anno_file} which has objects: {objects}")	

	logging.info('Creating a new image now...')
	
	for attempt in range(MAX_ATTEMPTS_TO_SYNTHESIZE):
		logging.info(f'\tStarting {attempt}th attempt to synthesize this image...')
		start_time = time.time()
		already_syn_objs = []
		all_objects_success = True
		objs_n_masks = [] # reset the list of objects and masks to try again
		# try to place each object that's been assigned to this image (or skip if unplaceable)
		for idx, obj in enumerate(objects):
			logging.info(f'\t\tStarting object {obj[0]}...')
			foreground = Image.open(obj[0]).convert('RGB')
			foreground = ImageOps.exif_transpose(foreground)  # fix orientation if needed
			mask_file =  os.path.join(os.path.dirname(obj[0]), os.path.basename(obj[0]).split('.')[0] + '_mask.png').replace('/images', '/masks')
			
			original_mask = Image.open(mask_file)
			mask_arr = PIL2array1C(original_mask)
			if INVERTED_MASK:
				mask = Image.fromarray(255-mask_arr).convert('1')
			else:
				mask = original_mask.convert('1')

			assert mask.size == foreground.size, f"Mask size {mask.size} does not match foreground size {foreground.size} for object {obj[0]}"
			
			o_w, o_h = foreground.size

			if rotation_augment:
				# logging.info(f'\t\tRotating object of size {o_w}x{o_h}...')
				rot_degrees = random.randint(-MAX_DEGREES, MAX_DEGREES)

				foreground = foreground.rotate(rot_degrees, expand=True)
				mask = mask.rotate(rot_degrees, expand=True)

				dilated_mask = Image.fromarray(cv2.dilate(PIL2array1C(mask), np.ones((20,20), np.uint8), iterations=1))#, 'L')
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
			
				foreground = foreground.crop((xmin, ymin, xmax, ymax))
				mask = mask.crop((xmin, ymin, xmax, ymax))
				dilated_mask = dilated_mask.crop((xmin_d, ymin_d, xmax_d, ymax_d))

				# logging.info(f'\t\t Foreground size after crop: {foreground.size} | Dilated foreground size after crop: {dilated_foreground.size}')
				# logging.info(f'\t\t Mask size after crop: {mask.size} | Dilated mask size after crop: {dilated_mask.size}')

				o_w, o_h = foreground.size
				logging.info(f'\t\t\tObject rotated by {rot_degrees} degrees, new size: {o_w}x{o_h}')

			if o_w < 20 or o_h < 20:
				logging.warning(f'\t\t\tObject (after crop n rotate) is quite small!! ({o_w}x{o_h})...')

			if scale_augment:
				ACTUAL_MIN_SCALE = MIN_SCALED_DIM / min(o_w, o_h) # every object should be at least MIN_SCALED_DIM pixels in width/height
				ACTUAL_MAX_SCALE = min(w,h) / max(o_w, o_h) *0.95 # every object should be at most min(w,h) pixels in width/height
				# the *0.95 ensures the object isn't exactly as wide/tall as the background since that may still give errors

				# scale the object so it occupies similar fraciton of the image as it did in the original foreground image
				length_scale = ((w*h) / (original_mask.size[0] * original_mask.size[1]))**0.5 
				scale = random.uniform(max(MIN_SCALE*length_scale, ACTUAL_MIN_SCALE), min(ACTUAL_MAX_SCALE, MAX_SCALE*length_scale))

				o_w, o_h = int(scale*o_w), int(scale*o_h)
				assert (0 < o_w < w) and (0 < o_h < h), "Invalid object dimensions after scaling"

				foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
				mask = mask.resize((o_w, o_h), Image.LANCZOS)
				dilated_mask = dilated_mask.resize((o_w, o_h), Image.LANCZOS)
				
			# Compare current mask with all previous masks to avoid excess occlusion
			xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
			for placement_attempt in range(MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE):
				logging.info(f'\t\t\tStarting {placement_attempt}-th object placement attempt...')
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
						logging.info('\t\t\t\tOcclusion found, trying again...')
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
			
			objs_n_masks.append((obj[1], (x, y, xmin, ymin, xmax, ymax), foreground, mask, dilated_mask))

		# If we could place all objects, then we're done trying object placements
		if all_objects_success:
			break        
	
	background = Image.open(bg_file).convert('RGB')
	background = ImageOps.exif_transpose(background)  # fix orientation if needed
	background = background.resize((w, h), Image.LANCZOS)

	synth_images = []
	for i in range(len(blending_list)):
		synth_images.append(background.copy())

	logging.info('\tBlending objects now...')
	# Start pasting and blending objects
	labels = []
	for idx, obj_n_mask in enumerate(objs_n_masks):
		obj_class, (x, y, xmin, ymin, xmax, ymax), foreground, mask, dilated_mask = obj_n_mask
		# Paste image on different background copies according to the different blending modes
		for i in range(len(blending_list)):
			if blending_list[i] == 'none' or blending_list[i] == 'motion':
				synth_images[i].paste(foreground, (x, y), mask)

			elif blending_list[i] == 'poisson':
				offset = (x, y)
				target = PIL2array3C(synth_images[i])

				source = PIL2array3C(foreground)
				source, dilated_mask_arr, offset = trim_img_n_mask(target, source, PIL2array1C(dilated_mask), offset)
				dilated_mask_arr = (dilated_mask_arr*255).astype(np.uint8)
				
				assert (offset[0] + source.shape[1] < w) and (offset[1] + source.shape[0] < h), \
					f"Offset {offset} with source shape {source.shape} exceeds target dimensions {target.shape}"
				
				center = (offset[0] + source.shape[1]//2, offset[1] + source.shape[0]//2)

				assert source.shape[0] > 0 and source.shape[1] > 0, f"Source image is empty for object {obj_class} at index {idx}"
				assert target.shape[0] > 0 and target.shape[1] > 0, f"Target image is empty for object {obj_class} at index {idx}"
				assert dilated_mask_arr.shape[0] > 0 and dilated_mask_arr.shape[1] > 0, f"Dilated mask is empty for object {obj_class} at index {idx}"
				assert source.shape[0] == dilated_mask_arr.shape[0] and source.shape[1] == dilated_mask_arr.shape[1], \
					f"Source and dilated mask shapes do not match for object {obj_class} at index {idx}: {source.shape} vs {dilated_mask_arr.shape}"
				
				mixed = cv2.seamlessClone(source.copy(), target.copy(), dilated_mask_arr, center, cv2.NORMAL_CLONE)

				synth_images[i] = Image.fromarray(mixed.astype(np.uint8))#, 'RGB')

			elif blending_list[i] == 'gaussian':
				synth_images[i].paste(foreground, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))

			elif blending_list[i] == 'box':
				synth_images[i].paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))

		# Save annotations in text file
		class_num = [key for key in label_map if label_map[key] == obj_class][0]
		xmin = max(1, x+xmin)
		xmax = min(w, x+xmax)
		ymin = max(1, y+ymin)
		ymax = min(h, y+ymax)
		string = f"{class_num} {(xmin+xmax)/(2*w)} {(ymin+ymax)/(2*h)} {(xmax-xmin)/w} {(ymax-ymin)/h}\n"
		labels.append(string)

	# write annotations to file
	logging.info(f'\tWriting annotations to {anno_file}')
	with open(anno_file, 'w') as f:
		f.writelines(labels)

	logging.info(f'\tCreating image with all objects took {time.time() - start_time} seconds')
	logging.info('\tSaving images now...')
	start_time = time.time()

	# Save images
	for i in range(len(blending_list)):
		# blend all objects at once for motion blur
		if blending_list[i] == 'motion':
			synth_images[i] = LinearMotionBlur3C(PIL2array3C(synth_images[i]))
		synth_images[i].save(img_file.replace('none', blending_list[i]))

	logging.info(f'Saving images took {time.time() - start_time} seconds')
   
def gen_syn_data(n_images, img_files, classes, img_dir, anno_dir, label_map, parallelize,
				 scale_augment, rotation_augment, allow_full_occlusion, add_distractors):
	'''Creates list of objects and distrctor objects to be pasted on what images.
	   Spawns worker processes and generates images according to given params

	Args:
		n_images(int): Number of images to generate
		img_files(list): List of image files
		classes(list): List of the object class of each image
		img_dir(str): Directory where synthesized images will be stored
		anno_dir(str): Directory where corresponding annotations will be stored
		label_map(dict): Mapping of labels to class names
		parallelize(bool): Whether to run the dataset generation in parallel mode
		scale_augment(bool): Add scale data augmentation
		rotation_augment(bool): Add rotation data augmentation
		allow_full_occlusion(bool): Generate images without checking for too much occlusion
		add_distractors(bool): Add distractor objects
	'''
	background_files = glob.glob(os.path.join(BACKGROUND_DIR, BACKGROUND_GLOB_STRING))
	logging.info(f"Number of background images : {len(background_files)}")

	objects = list(zip(img_files, classes))
	random.shuffle(objects)
	target_objects = [pair for pair in objects if pair[1] in SELECTED_CLASSES]
	distractor_objects = [pair for pair in objects if pair[1] not in SELECTED_CLASSES]

	assert len(target_objects) > 0, "No target objects found in the dataset"
	if add_distractors:
		assert len(distractor_objects) > 0, "No distractor objects found in the dataset"

	img_files = []
	anno_files = []
	params_list = []
	for img_idx in range(n_images):
		# Get list of objects
		selected_objects = []
		n_objects = random.randint(MIN_N_OBJECTS, MAX_N_OBJECTS)
		n_target_objects = random.randint(MIN_N_TARGET_OBJECTS, MAX_N_TARGET_OBJECTS)

		selected_objects.extend(random.sample(target_objects, n_target_objects))

		if add_distractors:
			selected_objects.extend(random.sample(distractor_objects, n_objects-n_target_objects))

		logging.info(f"Chosen objects: {selected_objects}")
		
		bg_file = random.choice(background_files)

		logging.info(f"Chosen background file: {bg_file}")
		
		img_file = os.path.join(img_dir, f'{img_idx}_none.jpg')
		anno_file = os.path.join(anno_dir, f'{img_idx}.txt')
		params_list.append((selected_objects, img_file, anno_file, bg_file, label_map)) 
		
	if parallelize:
		logging.info(f"Parallelizing with {NUMBER_OF_WORKERS} workers")
		partial_func = partial(create_image_anno_wrapper, w=WIDTH, h=HEIGHT, scale_augment=scale_augment, rotation_augment=rotation_augment,
						  blending_list=BLENDING_LIST, allow_full_occlusion=allow_full_occlusion) 
		p = Pool(NUMBER_OF_WORKERS, init_worker)
		try:
			p.map(partial_func, params_list)
		except KeyboardInterrupt:
			logging.warning("....\nCaught KeyboardInterrupt, terminating workers")
			p.terminate()
		else:
			p.close()
		p.join()
	else:
		logging.info("Not parallelizing, running in serial mode")
		for params in params_list:
			create_image_anno_wrapper(params, w=WIDTH, h=HEIGHT, scale_augment=scale_augment, rotation_augment=rotation_augment,
							  blending_list=BLENDING_LIST, allow_full_occlusion=allow_full_occlusion)

	print('Generation complete!')
	logging.info("Generation complete!")
	logging.info(f'Current datetime: {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}')
	
def init_worker():
	''' Catch Ctrl+C signal to terminate workers '''
	signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
	''' Generate synthetic dataset according to given args '''
	img_files = glob.glob(os.path.join(args.root, '*', 'images', '*'))[:args.max_obj_images]
	random.shuffle(img_files)
	class_names = [img_file.split('/')[-3] for img_file in img_files]

	unselected_classes = set(class_names) - set(SELECTED_CLASSES)
	all_classes = SELECTED_CLASSES + list(unselected_classes)
	label_map = {i: label for i, label in enumerate(all_classes)}

	if not os.path.exists(args.exp):
		os.makedirs(args.exp)

	anno_dir = os.path.join(args.exp, 'labels')
	img_dir = os.path.join(args.exp, 'images')

	if not os.path.exists(os.path.join(anno_dir)):
		os.makedirs(anno_dir)
	if not os.path.exists(os.path.join(img_dir)):
		os.makedirs(img_dir)
	
	gen_syn_data(args.n_images, img_files, class_names, img_dir, anno_dir, label_map, args.parallelize,
										   args.scale, args.rotation, args.allow_full_occlusion, args.add_distractors)
	write_yaml_file(args.exp, label_map)

	return label_map

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Create dataset with different augmentations",
									  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("root",
	  help="The root directory which contains the images and annotations.")
	parser.add_argument("exp",
	  help="The directory where images and annotation lists will be created.")
	parser.add_argument("--selected",
	  help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory", action="store_true")
	parser.add_argument("--no_scale",
	  help="Remove scale augmentation. Default is to add scale augmentation.", action="store_true")
	parser.add_argument("--no_rotation",
	  help="Remove rotation augmentation. Default is to add rotation data augmentation.", action="store_true")
	# parser.add_argument("--num",
	#   help="Number of times each image will be in dataset", default=1, type=int)
	parser.add_argument("--n_images",
	  help="Number of images to generate (divided by 5 for different blending modes)", default=10000, type=int)
	parser.add_argument("--allow_full_occlusion",
	  help="Allow complete occlusion between objects (faster). Default is to avoid high occlusions (as defined by MAX_OCCLUSION_IOU)", action="store_true")
	parser.add_argument("--no_distractors",
	  help="Don't add distractors objects. Default is to add distractors", action="store_true")
	parser.add_argument("--dont_parallelize",
	  help="Run the dataset generation in serial mode. Default is to run in parallel mode", action="store_true")
	parser.add_argument("--max_obj_images",
	  help="Maximum number of object images to use overall", default=int(1e9), type=int)
	args = parser.parse_args()

	args.parallelize = not args.dont_parallelize
	args.scale = not args.no_scale
	args.rotation = not args.no_rotation
	args.add_distractors = not args.no_distractors
	del args.no_scale
	del args.no_rotation
	del args.dont_parallelize
	del args.no_distractors

	if not os.path.exists(args.exp):
		os.makedirs(args.exp)
		
	logger = logging.getLogger(__name__)
	logging.basicConfig(filename=f"{args.exp}/dataset_generator.log", encoding='utf-8', level=logging.DEBUG)
	logger.setLevel(logging.INFO)
	logging.getLogger('PIL').setLevel(logging.WARNING)
	logging.info(f'Generating dataset with scale={args.scale}, rotation={args.rotation}, n_images={args.n_images}, allow_full_occlusion={args.allow_full_occlusion}, add_distractors={args.add_distractors}')
	logging.info(f'Current datetime: {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}')

	# Generate the dataset
	label_map = generate_synthetic_dataset(args)
	reverse_label_map = {v: k for k, v in label_map.items()}
	post_processing.main(dataset_path=args.exp, classes_to_keep=[reverse_label_map[cls] for cls in SELECTED_CLASSES])