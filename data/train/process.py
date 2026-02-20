import cv2
import os

def preprocess_image(image_path, size=(128,128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = img / 255.0  # normalize
    return img
