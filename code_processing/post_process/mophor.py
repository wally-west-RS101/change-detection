import rasterio
import cv2
import numpy as np
import os , glob 

def Morphology(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # dilation  
    # img = cv2.dilate(data,kernel,iterations = 1)
    # opening
    img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # for i in range(10):
    #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    # closing
    #     for _ in range(2):
    # img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    
    return img


def delete_cloud_shadow(img):
    img[img != 1] = 0
    return img