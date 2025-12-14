# person2_preprocessing.py

from PIL import Image
import numpy as np
import cv2
import os

# Load a single image
def load_image(path):
    """
    Load an image from the given file path.
    Returns a NumPy array.
    """
    img = Image.open(path)
    return np.array(img)

# Convert image to grayscale
def to_gray(img_array):
    """
    Convert an RGB image array to grayscale.
    If the image is already grayscale, it returns as is.
    """
    if len(img_array.shape) == 3:  # RGB image
        return np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    return img_array

# Apply histogram equalization
def apply_histogram_equalization(gray_img):
    """
    Enhance the contrast of a grayscale image using histogram equalization.
    Input must be a grayscale image (2D array).
    """
    gray_uint8 = gray_img.astype('uint8')
    return cv2.equalizeHist(gray_uint8)

# Load and preprocess all images in a folder
def preprocess_folder(folder_path):
    """
    Load, convert to grayscale, and equalize all images in a folder.
    Returns a list of processed image arrays.
    """
    processed_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img_path = os.path.join(folder_path, filename)
            img = load_image(img_path)
            gray = to_gray(img)
            equalized = apply_histogram_equalization(gray)
            processed_images.append(equalized)
    return processed_images
