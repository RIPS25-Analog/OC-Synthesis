# Paths
# Fill this according to own setup
BACKGROUND_DIR = '/home/data/raw/backgrounds/bg-20k/BG-20k/train'
BACKGROUND_GLOB_STRING = '*.jpg'
INVERTED_MASK = True # Set to true if white pixels represent background
SELECTED_CLASSES = ['hammer', 'screwdriver', 'wrench'] # these classes get labelles as 0,1,...

# Parameters for generator
NUMBER_OF_WORKERS = 24
BLENDING_LIST = ['gaussian','poisson', 'none', 'box', 'motion']
TRAIN_VAL_TEST_SPLIT = [1.0, 0.0, 0.0]

# Parameters for images
MIN_NO_OF_OBJECTS = 1
MAX_NO_OF_OBJECTS = 4
WIDTH = 640*2
HEIGHT = 480*2
MAX_ATTEMPTS_TO_SYNTHESIZE = 5
MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE = 10

# Parameters for objects in images
MIN_SCALE = 0.08 # min scale for scale augmentation
MAX_SCALE = 0.35 # max scale for scale augmentation
MIN_SCALED_DIM = 2 # minimum scaled width/height of object (in pixels) after scale augmentation
MAX_DEGREES = 180 # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = 0.25 # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
MAX_OCCLUSION_IOU = 0.75 # IOU > MAX_OCCLUSION_IOU is considered an occlusion
MIN_WIDTH = 6 # Minimum width of object to use for data generation
MIN_HEIGHT = 6 # Minimum height of object to use for data generation