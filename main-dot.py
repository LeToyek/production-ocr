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
    image = cv2.imread('cap-full-text.jpeg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('cap-full-text-blue.jpeg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('kardus.jpeg', cv2.IMREAD_GRAYSCALE)

    # Set kernel (structuring element) size
    kernel_size = 3
    max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Thresholding
    thresh = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY_INV)[1]

    # Set iteration ranges for dilate, erode, and closing
    dilate_iterations_range = range(4, 10)
    erode_iterations_range = range(1, 7)
    closing_iterations_range = range(1, 5)

    best_dilate_iterations = 0
    best_erode_iterations = 0
    best_closing_iterations = 0
    best_text = ""
    best_text_count = 0

    for dilate_iter in dilate_iterations_range:
        # Dilate
        dilate = cv2.dilate(thresh, max_kernel, iterations=dilate_iter)
        
        for erode_iter in erode_iterations_range:
            # Erode
            kernel_erode = np.ones((3, 3), np.uint8)
            erode = cv2.erode(dilate, kernel_erode, iterations=erode_iter)
           
           # Use pytesseract to extract text
            custom_config = r'lang="dotslayer.traineddata" --oem 3 --psm 6'
            # text = pytesseract.image_to_string(closing_image, config=tessdata_dir_config)
            text = pytesseract.image_to_string(erode, config=custom_config)
            print("Dilate iterations:", dilate_iter)
            print("Erode iterations:", erode_iter)
            # print("Closing iterations:", closing_iter)
            print("Text:", text)
            time.sleep(1)

    print("Best dilate iterations:", best_dilate_iterations)
    print("Best erode iterations:", best_erode_iterations)
    print("Best closing iterations:", best_closing_iterations)
    print("Extracted text:", best_text)


    # Display the images
    plt.figure(figsize=(10,5))

    plt.subplot(1,5,1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1,5,2)
    plt.title('Threshold Image')
    plt.imshow(thresh, cmap='gray')
    
    plt.subplot(1,5,3)
    plt.title('Dilate Image')
    plt.imshow(dilate, cmap='gray')
    
    plt.subplot(1,5,4)
    plt.title('Erode Image')
    plt.imshow(erode, cmap='gray')

    # plt.subplot(1,5,5)
    # plt.title('Final Image (closing)')
    # plt.imshow(closing_image, cmap='gray')

    plt.show()