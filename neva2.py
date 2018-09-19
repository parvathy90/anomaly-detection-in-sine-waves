## Importing the necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("C:/Users/user/Downloads/ML_Assignment_Neva Innovation_Jul_18/10^-1"))

# Any results you write to the current directory are saved as output.
img = cv2.imread('C:/Users/user/Downloads/ML_Assignment_Neva Innovation_Jul_18/10^-1/IMG_3134.JPG')
#plt.imshow(img)
## take this image as the input to our first function

def extract_white_strip(image):
    ## first convert the image format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
     # HSV separates luma, or the image intensity, from chroma or the color information.
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    ## convert RGB to gray
    ##image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ## creating the mask
    mini = np.array([255,255,0])
    maxi = np.array([255,255, 102])
    mask1 = cv2.inRange(image_hsv, mini, maxi)

    
    ## Thresholding the image
    ret, thresh2 = cv2.threshold(image_hsv, 127, 255, cv2.THRESH_BINARY)

    
    
    
    ## find the biggest rectangle
    biggest_contour = find_biggest_contour(thresh2)
    
    overlay = overlay_mask(mask1, image)
    
    ## 
    final_image , dimension = rectangle_contour(overlay, biggest_contour)
    print(dimension)
    
    plt.figure()
    frame1 = plt.gca()
    plt.imshow(final_image)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.show()  # display it
def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    
    return img
def find_biggest_contour(image):
    # Copy
    image = image.copy()
    ## converting the image in the proper form 
    rgbimg = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    the = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
    
    we, contours, hierarchy = cv2.findContours(the, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    
    return biggest_contour
def rectangle_contour(image, contour):
    image_rect = image.copy()
    x,y,w,h = cv2.boundingRect(contour)
    image_rect = cv2.rectangle(image_rect, (x,y), (x+w,y+h), (0,255,0), 2)
    ## extract the white strip
    crop_img = image_rect[y:y+h, x:x+w]
    dimension = (x,y,x+w,y+h)
    return crop_img , dimension
extract_white_strip(img)	