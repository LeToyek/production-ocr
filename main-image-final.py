import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter


def image_preprocessor(image):
  """Preprocesses the input image for text extraction."""
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur to reduce noise
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  # Set kernel size for morphological operations
  kernel_size = 3
  max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

  # Adaptive thresholding for uneven illumination
  thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

  # Perform morphological operations to refine text regions
  kernel_erode = np.ones((3, 3), np.uint8)
  dilate = cv2.dilate(thresh, max_kernel, iterations=8)
  erode = cv2.erode(dilate, kernel_erode, iterations=5)
  closing_image = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, max_kernel, None, None, 1, cv2.BORDER_REFLECT101)

  return thresh

def extract_text(image):
  """Extracts text from the preprocessed image."""
  preprocessed_image = image_preprocessor(image)
  custom_config = r'--oem 3 --psm 6 --psm 10 tessedit_char_whitelist=0123456789KLmIndexMtr'  # Refined char_whitelist
  pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
  text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
  return text, preprocessed_image

def main():
  """Loads the image, extracts text, and displays results."""
  image = cv2.imread('cap-full-text-blue.jpeg')  # Replace with your image path

  text,preprocessed_image = extract_text(image.copy())
  print("Extracted Text:", text)

  # Display the images (optional)
  plt.figure(figsize=(15, 8))

  plt.subplot(2, 3, 1)
  plt.title('Original Image')
  plt.imshow(image, cmap='gray')

  plt.subplot(2, 3, 2)
  plt.title('Preprocessed Image')
  plt.imshow(preprocessed_image, cmap='gray')

  plt.subplot(2, 3, 3)
  plt.title('Extracted Text')
  plt.text(0.5, 0.5, text, fontsize=18, ha='center', va='center')
  plt.axis('off')

  plt.show()

if __name__ == '__main__':
  main()
