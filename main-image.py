import time

import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter
from ultralytics import YOLO

from app import create_app

app = create_app()

if __name__ == '__main__':
    traineddata_path = "./dotslayer.traineddata"
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    tessdata_dir_config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.,:mIndexMtr'

    # Load the YOLO model (download if not already available)
    # model = YOLO('yolov5s.pt')  # Using a small YOLOv5 model for example

    # image = cv2.imread('testocr.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('cap-full-text.jpeg', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('cap-full-text-blue.jpeg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('kardus.jpeg', cv2.IMREAD_GRAYSCALE)

    # Set kernel (structuring element) size
    kernel_size = 3
    max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Thresholding
    thresh = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY_INV)[1]

    # Set iteration ranges for dilate, erode, and closing
    
  
    kernel_erode = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(thresh, max_kernel, iterations=8)
    erode = cv2.erode(dilate, kernel_erode, iterations=5)
    closing_image = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, max_kernel, None, None, 1, cv2.BORDER_REFLECT101)
    custom_config = r'lang="dotslayer.traineddata" --oem 3 --psm 6'
    text = pytesseract.image_to_string(closing_image, config=custom_config)
    print("Text:", text)
    text_count = len(text)


    # Display the images
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 5, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 5, 2)
    plt.title('Threshold Image')
    plt.imshow(thresh, cmap='gray')

    plt.subplot(2, 5, 3)
    plt.title('Dilate Image ({} iterations)'.format(8))
    plt.imshow(dilate, cmap='gray')

    plt.subplot(2, 5, 4)
    plt.title('Erode Image ({} iterations)'.format(5))
    plt.imshow(erode, cmap='gray')

    plt.subplot(2, 5, 5)
    plt.title('Final Image (Closing) ({} iterations)'.format(1))
    plt.imshow(closing_image, cmap='gray')

    plt.subplot(1, 5,3)
    plt.text(0.5, 0.5, 'Best Extracted Text:\n{}'.format(text), fontsize=12, ha='center')
    plt.axis('off')

    plt.show()