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
    inputImage = cv2.imread('cap-full-text.jpeg')
    # image = cv2.imread('cap-full-text-blue.jpeg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('kardus.jpeg', cv2.IMREAD_GRAYSCALE)
    
    inputCopy = inputImage.copy()
    
    image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Set kernel (structuring element) size
    kernel_size = 3
    max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Thresholding
    # Set the adaptive thresholding (gasussian) parameters:
    windowSize = 31
    windowConstant = -1
    # Apply the threshold:
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, windowConstant)

    # Set iteration ranges for dilate, erode, and closing
    
        # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
    cv2.connectedComponentsWithStats(thresh, connectivity=4)

    # Set the minimum pixels for the area filter:
    minArea = 20

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')
    
  
    kernel_erode = np.ones((3, 3), np.uint8)
    # dilate = cv2.dilate(thresh, max_kernel, iterations=8)
    dilate = filteredImage
    erode = cv2.erode(dilate, kernel_erode, iterations=5)
    closing_image = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, max_kernel, None, None, 1, cv2.BORDER_REFLECT101)
    
    contours, hierarchy = cv2.findContours(closing_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []

    # Alright, just look for the outer bounding boxes:
    for i, c in enumerate(contours):

        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))


    # Draw the bounding boxes on the (copied) input image:
    for i in range(len(boundRect)):
        color = (0, 0, 255)
        cv2.rectangle(closing_image, (int(boundRect[i][0]), int(boundRect[i][1])), \
                (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    
    final_image = closing_image
    custom_config = r'lang="dotslayer.traineddata" --oem 3 --psm 6'
    text = pytesseract.image_to_string(final_image, config=custom_config)
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
    plt.imshow(final_image, cmap='gray')

    plt.subplot(1, 5,3)
    plt.text(0.5, 0.5, 'Best Extracted Text:\n{}'.format(text), fontsize=12, ha='center')
    plt.axis('off')

    plt.show()