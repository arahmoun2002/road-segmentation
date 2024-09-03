import os
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple, Union
from scipy.ndimage import gaussian_filter, laplace
from PIL import ImageEnhance


BATCH_SIZE = 16
HEIGHT = 224
WIDTH = 224
PIXELS = 255
TRAIN_PATH = "/home/ryounis/Documents/Zurich/ETHZ/MA2/CIL/Road-Segmentation-Comp/ethz-cil-road-segmentation-2024/training"


def create_preprocessed_dataset(path: str = TRAIN_PATH, val_size: float = 0.2, batch_size: int = BATCH_SIZE):
    """
    Creates a preprocessed dataset instance from the specified path.

    Args:
        path (str, optional): The path to the /training directory. Defaults to TRAIN_PATH.
        val_size (float, optional): The size of the validation set. Defaults to 0.2.

    Returns:
        tuple[tf.data.DatasetV2, tf.data.DatasetV2]: A tuple containing two TensorFlow datasets: the training dataset and the validation dataset.
    """
    (train_x, train_y), (val_x, val_y) = _load_data(path, val_size)
    train_dataset = _tf_dataset(train_x, train_y, batch_size=batch_size)
    val_dataset = _tf_dataset(val_x, val_y, batch_size=batch_size)
    return train_dataset, val_dataset

def get_preprocessed_data(path: str = TRAIN_PATH, val_size: float = 0.2):
    """Load and preprocess the data.

    Args:
        path (str, optional): The path to the data. Defaults to TRAIN_PATH.
        val_size (float, optional): The proportion of data to use for validation. Defaults to 0.2.

    Returns:
        Tuple[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]]]: 
        A tuple containing the preprocessed training data and the preprocessed validation data.
        The training data is represented as a tuple of lists, where the first list contains the preprocessed 
        training images and the second list contains the corresponding ground truth labels.
        The validation data is represented in the same way.
    """
    (train_x, train_y), (val_x, val_y) = _load_data(path, val_size)
    train_x = np.array([_read_image(x) for x in train_x])
    train_y = np.array([_read_gt(y) for y in train_y])
    val_x = np.array([_read_image(x) for x in val_x])
    val_y = np.array([_read_gt(y) for y in val_y])
    print("CHECK")
    train_x, train_y = augment_images_and_labels(train_x, train_y)
    aug_val_x, aug_val_y = augment_images_and_labels(val_x, val_y)
    
    return (train_x, train_y), (aug_val_x, aug_val_y), (val_x, val_y)

def get_deepglobe_data(path, N_FILES=1000):
    # List all files in the directory
    print("Listing files")
    all_files = os.listdir(path)
    
    # Filter out only the satellite images
    print("Filtering files")
    image_files = [file for file in all_files if file.endswith('.jpg')]
    
    # Choose N_FILES random satellite images
    print("Choosing N_FILES")
    image_files = [path + p for p in np.random.choice(image_files, min(N_FILES, len(image_files)), replace=False)]
    
    # Get corresponding ground truth files
    print("Getting masks")
    ground_truth_files = [file.replace('_sat.jpg', '_mask.png') for file in image_files]
    ## Preprocessing files
    print("Preprocessing files")
    (train_x, train_y), (val_x, val_y) = preprocess_split_data(image_files, ground_truth_files)
        
    return train_x, train_y, val_x, val_y

def get_google_data(path, N_FILES=1000):
    # List all files in the directory
    print("Listing files")
    files = sorted(os.listdir(path + 'images/'))
    
    # Choose N_FILES random satellite images
    print("Choosing N_FILES")
    files = [p for p in np.random.choice(files, min(N_FILES, len(files)), replace=False)]
    
    # Get corresponding ground truth files
    print("Getting masks")
    image_files = [path + 'images/' + file for file in files]
    groundtruth_files = [path + 'groundtruth/' + file for file in files]
        
    ## Preprocessing files
    print("Preprocessing files")
    (train_x, train_y), (val_x, val_y) = preprocess_split_data(image_files, groundtruth_files)
        
    return train_x, train_y, val_x, val_y

def get_massa_data(path, N_FILES=1000):
    # List all files in the directory
    print("Listing files")
    files = sorted(os.listdir(path + 'train/'))
    
    # Choose N_FILES random satellite images
    print("Choosing N_FILES")
    files = [p for p in np.random.choice(files, min(N_FILES, len(files)), replace=False)]
    
    # Get corresponding ground truth files
    print("Getting masks")
    image_files = [path + 'train/' + file for file in files]
    groundtruth_files = [path + 'train_labels/' + file[:-1] for file in files]
        
    ## Preprocessing files
    print("Preprocessing files")
    (train_x, train_y), (val_x, val_y) = preprocess_split_data(image_files, groundtruth_files)
        
    return train_x, train_y, val_x, val_y

def augment_images_and_labels(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment the given images and labels with horizontal flips, vertical flips, and affine transformations.

    Args:
        images (np.ndarray): Array of images with shape (N, H, W, C).
        labels (np.ndarray): Array of labels with shape (N, H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of augmented images and labels.
    """
    augmented_images = []
    augmented_labels = []

    for img, lbl in zip(images, labels):
        orig_img = img
        orig_lbl = lbl
        augmented_images.append(orig_img)
        augmented_labels.append(orig_lbl)
        
        
        # Horizontal flip
        hor_flip_img = np.fliplr(orig_img)
        hor_flip_lbl = np.fliplr(orig_lbl)
        augmented_images.append(hor_flip_img)
        augmented_labels.append(hor_flip_lbl)

        # Vertical flip
        ver_flip_img = np.flipud(orig_img)
        ver_flip_lbl = np.flipud(orig_lbl)
        augmented_images.append(ver_flip_img)
        augmented_labels.append(ver_flip_lbl)

        # Horizontal and vertical flip
        hor_ver_flip_img = np.fliplr(np.flipud(orig_img))
        hor_ver_flip_lbl = np.fliplr(np.flipud(orig_lbl))
        augmented_images.append(hor_ver_flip_img)
        augmented_labels.append(hor_ver_flip_lbl)

        # Affine transformation
        rows, cols = img.shape[:2]
        src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        dst_points = np.float32([[cols * 0.1, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.2, rows * 0.9]])
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        affine_img = cv2.warpAffine(img, affine_matrix, (cols, rows))
        affine_lbl = np.expand_dims(cv2.warpAffine(lbl, affine_matrix, (cols, rows)), axis=-1)

        ## Print all added images and labels shapes
        # print("Added images and labels shapes:")
        # print(orig_img.shape, orig_lbl.shape, hor_flip_img.shape, hor_flip_lbl.shape, ver_flip_img.shape, ver_flip_lbl.shape, hor_ver_flip_img.shape, hor_ver_flip_lbl.shape, affine_img.shape, affine_lbl.shape)        
        augmented_images.append(affine_img)
        augmented_labels.append(affine_lbl)

    return np.array(augmented_images), np.array(augmented_labels)
    
def preprocess_split_data(images, groundtruths):
    """Preprocess the data and separate the images and their ground truth into train val splits.
    """
    print("Opening cropped images...")
    processed_images_and_masks = [
        _crop_image_and_mask(Image.open(x).convert('RGB'), Image.open(y).convert('L'))
        for x, y in zip(images[:len(images) // 2], groundtruths[:len(groundtruths) // 2])
    ]
    
    print("Opening resized images...")
    resized_images_and_masks = [
        (Image.open(x).convert('RGB').resize((WIDTH, HEIGHT)), Image.open(y).convert('L').resize((WIDTH, HEIGHT)))
        for x, y in zip(images[len(images) // 2:], groundtruths[len(groundtruths) // 2:])
    ]
    
    cropped_images, cropped_groundtruths = zip(*processed_images_and_masks)
    resized_images, resized_groundtruths = zip(*resized_images_and_masks)
    
    images = np.concatenate([np.array([np.array(img) for img in resized_images]), np.array([np.array(img) for img in cropped_images])])
    groundtruths = np.concatenate([np.array([np.array(mask) for mask in resized_groundtruths]), np.array([np.array(mask) for mask in cropped_groundtruths])])
    
    # Preprocess the images and groundtruths
    print("Preprocessing images...")
    images = normalize_pixels(images)
    groundtruths = normalize_pixels(groundtruths)
    
    # Split the data into training and validation sets
    print("Splitting data into training and validation sets...")
    train_x, val_x, train_y, val_y = train_test_split(images, groundtruths, test_size=0.1, random_state=42)
    
    return (train_x, np.expand_dims(train_y, axis=-1)), (val_x, np.expand_dims(val_y, axis=-1))

###########################################
############ Private functions ############
###########################################

def _load_data(path: str = TRAIN_PATH, val_size: float = 0.1) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
    """Load the data from the specified path.

    Args:
        path (str, optional): The path to the data directory. Defaults to TRAIN_PATH.
        val_size (float, optional): The size of the validation set. Defaults to 0.2.

    Returns:
        tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]: A tuple containing two tuples. The first tuple contains two lists of file paths: the paths to the input images and the paths to the corresponding groundtruth images for training. The second tuple contains two lists of file paths: the paths to the input images and the paths to the corresponding groundtruth images for validation.
    """
    train_x = sorted([f'{path}/images/{filename}' for filename in os.listdir(f'{path}/images')  if filename.endswith('.png')])
    train_y = sorted([f'{path}/groundtruth/{filename}' for filename in os.listdir(f'{path}/groundtruth')  if filename.endswith('.png')])
    
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_size, random_state=42)

    return (train_x, train_y), (val_x, val_y)


def _crop_image_and_mask(img, mask):    
    # Ensure the image and the mask have the same size
    assert img.size == mask.size, "The image and the mask must have the same size"
    
    # Get the dimensions of the image
    img_width, img_height = img.size
    
    # Initialize variables to hold the best crop and its non-black pixel count
    best_cropped_image = None
    best_cropped_mask = None
    max_non_black_pixels = -1
    
    # Loop over the image and mask, and crop
    for top in range(0, img_height, int(HEIGHT / 4)):
        for left in range(0, img_width, int(WIDTH / 4)):
            # Define the box to crop
            box = (left, top, left + WIDTH, top + HEIGHT)
            
            # Crop the image and the mask
            cropped_image = img.crop(box)
            cropped_mask = mask.crop(box)
            
            # Convert cropped mask to numpy array to check for non-black pixels
            mask_array = np.array(cropped_mask)
                        # Count non-black (non-zero) pixels in the mask
            non_black_pixel_count = np.count_nonzero(mask_array)
            
            # Update the best crop if the current one has more non-black pixels
            if non_black_pixel_count > max_non_black_pixels:
                best_cropped_image = cropped_image
                best_cropped_mask = cropped_mask
                max_non_black_pixels = non_black_pixel_count
    
    return best_cropped_image, best_cropped_mask 

def _read_image(path: Union[str, bytes]) -> np.ndarray:
    if type(path) == 'bytes':
        path = path.decode()
    img = Image.open(path).convert('RGB')  # Ensure RGB format
    img = img.resize((WIDTH, HEIGHT))
    x = np.array(img, dtype=np.float32) / 255.0  # Normalize
    return x

def _read_gt(path: Union[str, bytes]) -> np.ndarray:
    """
    Read a groundtruth image from the given path and preprocess it.
    This function is only supposed to be called by a numpy_function

    Args:
        path (bytes): The path to the groundtruth image file.

    Returns:
        np.ndarray: The preprocessed groundtruth image as a numpy array.
    """
    if type(path) == 'bytes':
        path = path.decode()
    y = Image.open(path).convert('L')  # Convert to grayscale
    y = y.resize((WIDTH, HEIGHT))
    y = np.expand_dims(y, axis=-1)
    y = normalize_pixels(y)
    return y

def _tf_parse(x, y):
    def _parse(x: bytes, y: bytes) -> tuple[np.ndarray, np.ndarray]:
        x = _read_image(x)
        y = _read_gt(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([HEIGHT, WIDTH, 4])
    y.set_shape([HEIGHT, WIDTH, 1])
    
    return x, y

def _tf_dataset(x: list[str], y: list[str], batch_size: int = BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(_tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



###########################################
############ Image modifications functions ############
###########################################


def increase_contrast(image, factor=2.0):
    return ImageEnhance.Contrast(image).enhance(factor) 

def apply_gaussian_filter(image, sigma=1):
    return gaussian_filter(image, sigma=sigma)

def apply_laplace_filter(image):
    return laplace(image)

def apply_threshold(image, threshold=175):
    # Grayscale
    image = image.convert('L')
    # Threshold
    image = image.point( lambda p: 255 if p > threshold else 0 )
    # To mono
    image = image.convert('1')
    
    return image

def detect_shadows(img):
    # Convert to HSV color space
    if type(img) is not np.ndarray:
        img = np.array(img)
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for shadow colors (adjust as needed)
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([180, 255, 100])

    # Create a mask for shadow regions
    mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

    return mask

def remove_shadow(img, mask):
  # Invert the mask
  inv_mask = 255 - mask

  # Convert masks to float32 for division
  inv_mask = inv_mask.astype(np.float32) / 255

  # Multiply the image with the inverted mask
  img_no_shadow = img * inv_mask[:, :, np.newaxis]

  # Adjust brightness (optional)
  img_no_shadow = cv2.convertScaleAbs(img_no_shadow, alpha=1.5, beta=0)

  return img_no_shadow

def normalize_pixels(image):
    return image / 255.0