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
    tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'

    # Load the YOLO model (download if not already available)
    # model = YOLO('yolov5s.pt')  # Using a small YOLOv5 model for example

    # image = cv2.imread('testocr.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('cap-full-text.jpeg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('cap-full-text-blue.jpeg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('kardus.jpeg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('kardus-2.jpeg', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('./data/kardus/kardus-3.jpeg', cv2.IMREAD_GRAYSCALE)

    # Set kernel (structuring element) size
    kernel_size = 3
    max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    blurImage = cv2.GaussianBlur(image, (7, 7), 0)
    
    denoisedImage = cv2.fastNlMeansDenoising(blurImage, None, 15, 7, 21)
    
    # Thresholding
    thresh = cv2.threshold(denoisedImage, 127 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(
    denoisedImage,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or cv2.ADAPTIVE_THRESH_MEAN_C
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=29,  # Size of a pixel neighborhood used to calculate the threshold value
    C=2  # Constant subtracted from the mean or weighted mean
)
    
    # Invers
    # invertedImage = cv2.bitwise_not(thresh)
    
    finalImage = thresh
    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(finalImage, config=custom_config)
    print("Text:", text)
    text_count = len(text)
    
    dilate_iterations_range = range(4, 8)
    erode_iterations_range = range(1, 5)
    
    for dilate_iter in dilate_iterations_range:
        # Dilate
        dilate = cv2.dilate(finalImage, max_kernel, iterations=dilate_iter)
        
        for erode_iter in erode_iterations_range:
            # Erode
            erode = cv2.erode(dilate, max_kernel, iterations=erode_iter)
            
            # Use pytesseract to extract text
            custom_config = r'--psm 6'
            sharpen = cv2.GaussianBlur(erode, (0, 0),3 )
            text = pytesseract.image_to_string(sharpen, config=custom_config)
            print("Dilate Iterations:", dilate_iter)
            print("Erode Iterations:", erode_iter)
            print("Text:", text)
            text_count = len(text)
            
            # the preprocessed images
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 3, 1)
            plt.title('Grayscale Image')
            plt.imshow(image, cmap='gray')

            plt.subplot(2, 3, 2)
            plt.title('Binary Image')
            plt.imshow(denoisedImage, cmap='gray')

            plt.subplot(2, 3, 3)
            plt.title('Thresh Binary Image')
            plt.imshow(thresh, cmap='gray')
            
            plt.subplot(2, 3, 4)
            plt.title('dilate Image')
            plt.imshow(dilate, cmap='gray')
            
            plt.subplot(2, 3, 5)
            plt.title('Erode Image')
            plt.imshow(erode, cmap='gray')
            
            plt.subplot(2, 3, 6)
            plt.title('Sharpen Image')
            plt.imshow(sharpen, cmap='gray')
            
            plt.show()
    
def psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal
        return 100
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value
