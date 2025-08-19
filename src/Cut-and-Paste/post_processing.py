import random
import shutil
import os
import glob
import defaults

def filter_bboxes_classes_and_add_suffixes(source_dir, output_dir, classes_to_keep):
	'''filter the labels in the source_dir to only include bounding boxes of certain classes, and write these labels to output_dir'''
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	for label_file in glob.glob(os.path.join(source_dir, '*.txt')):
		with open(label_file, 'r') as f:
			lines = f.readlines()
		
		filtered_lines = []
		for line in lines:
			parts = line.strip().split()
			if len(parts) > 0 and int(parts[0]) in classes_to_keep:
				filtered_lines.append(line)
		
		# Write the filtered bounding boxes to a new file
		for suffix in defaults.BLENDING_LIST:
			output_label_file = label_file.replace(source_dir, output_dir).replace('.txt', f'_{suffix}.txt')
			if not os.path.exists(output_label_file):
				with open(output_label_file, 'w') as f:
					f.writelines(filtered_lines)
			else:
				print(f"Skipping {output_label_file} as it already exists.")

def split_cnp_into_train_val_test(source_dir, dest_parent_dir=None, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
	"""
	Splits the images and labels in the source directory into train, val, and test sets.
	The source directory should contain 'images' and 'labels' subdirectories.
	"""
	if dest_parent_dir is None:
		dest_parent_dir = source_dir
	if os.path.exists(dest_parent_dir):
		raise FileExistsError(f"Destination parent directory {dest_parent_dir} already exists. Not risking the overwrite.")
	print(f"Splitting dataset in {source_dir} to {dest_parent_dir}")

	images_dir = os.path.join(source_dir, 'images')
	labels_dir = os.path.join(source_dir, 'labels')

	# Create output directories (raise error if they already exist)
	for split in ['train', 'val', 'test']:
		os.makedirs(os.path.join(dest_parent_dir, split, 'images'), exist_ok=False)
		os.makedirs(os.path.join(dest_parent_dir, split, 'labels'), exist_ok=False)

	images = sorted(glob.glob(os.path.join(images_dir, '*.png'))+glob.glob(os.path.join(images_dir, '*.jpg')))
	image_ids = list(set([os.path.basename(img).split('_')[0] for img in images]))

	random.seed(0)
	random.shuffle(image_ids)
	
	train_size = int(len(image_ids) * train_ratio)
	val_size = int(len(image_ids) * val_ratio)
	test_size = int(len(image_ids) * test_ratio)
	print(f'Total unique images {len(image_ids)}, train: {train_size}, val: {val_size}, test: {test_size}')
	
	train_image_ids = image_ids[:train_size]
	val_image_ids = image_ids[train_size:train_size + val_size]
	test_image_ids = image_ids[train_size + val_size:train_size + val_size + test_size]

	for img in images:
		img_id = os.path.basename(img).split('_')[0]
		label = os.path.join(labels_dir, os.path.basename(img).replace('.png', '.txt').replace('.jpg', '.txt'))

		if img_id in train_image_ids:
			split = 'train'
		elif img_id in val_image_ids:
			split = 'val'
		elif img_id in test_image_ids:
			split = 'test'
		else:
			continue

		dest_dir = os.path.join(dest_parent_dir, split, 'images')
		label_dest_dir = os.path.join(dest_parent_dir, split, 'labels')

		shutil.copy(img, os.path.join(dest_dir, os.path.basename(img)))
		shutil.copy(label, os.path.join(label_dest_dir, os.path.basename(label)))

def main(dataset_path, classes_to_keep):
	# rename the folder at dataset_final_root to dataset_unsplit_root
	dataset_final_root = dataset_path
	dataset_unsplit_root = dataset_path + '-unsplit'
	if os.path.exists(dataset_final_root):
		print(f"Renaming {dataset_final_root} to {dataset_unsplit_root}")
		os.rename(dataset_final_root, dataset_unsplit_root)

	# rename the labels folder to labels-orig
	if os.path.exists(os.path.join(dataset_unsplit_root, 'labels')):
		print(f"Renaming {os.path.join(dataset_unsplit_root, 'labels')} to {os.path.join(dataset_unsplit_root, 'labels-orig')}")
		os.rename(os.path.join(dataset_unsplit_root, 'labels'), os.path.join(dataset_unsplit_root, 'labels-orig'))

	filter_bboxes_classes_and_add_suffixes(os.path.join(dataset_unsplit_root, 'labels-orig'), os.path.join(dataset_unsplit_root, 'labels'), classes_to_keep)

	images = sorted(glob.glob(os.path.join(dataset_unsplit_root, 'images', '*.png'))+glob.glob(os.path.join(dataset_unsplit_root, 'images', '*.jpg')))
	labels = sorted(glob.glob(os.path.join(dataset_unsplit_root, 'labels', '*.txt')))

	assert len(images) == len(labels), f"Number of images ({len(images)}) does not match number of labels ({len(labels)})"
	print(f"Number of images and labels match: {len(images)}")

	split_cnp_into_train_val_test(dataset_unsplit_root, 
						dest_parent_dir=dataset_final_root,
						train_ratio = 0.7, val_ratio = 0.2, test_ratio = 0.1)
	
	# check if the images and labels subdirectory in dataset_final_root have the same number of files
	for split in ['train', 'val', 'test']:
		images_split = sorted(glob.glob(os.path.join(dataset_final_root, split, 'images', '*.png'))+\
							  glob.glob(os.path.join(dataset_final_root, split, 'images', '*.jpg')))
		labels_split = sorted(glob.glob(os.path.join(dataset_final_root, split, 'labels', '*.txt')))
		assert len(images_split) == len(labels_split), f"Number of images in {split} set ({len(images_split)}) does not match number of labels ({len(labels_split)})"
		print(f"Number of images and labels in {split} set match: {len(images_split)}")

if __name__ == "__main__":
	dataset_path = '/home/data/processed/cnp-pace/toycar_can_v0'
	classes_to_keep = [0, 1]
	main(dataset_path, classes_to_keep)