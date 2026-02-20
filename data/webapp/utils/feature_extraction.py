import cv2

def extract_edges(image):
    return cv2.Canny(image, 100, 200)
