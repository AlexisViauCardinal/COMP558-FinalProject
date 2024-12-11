import numpy as np
import cv2 as cv
from skimage.measure import label, regionprops

__default_kernel = 7
__default_iteration = 1
__default_median_blur = 21

def dilate(image : np.ndarray, kernel : int = 7, iteration : int = 1) -> np.ndarray:
    '''
        Apply OpenCV's dilate with ones fillter
    '''

    kernel = np.ones((kernel, ) * 2, np.uint8)
    return cv.dilate(image, kernel, iterations = iteration)

def erode(image : np.ndarray, kernel : int = 7, iteration : int = 1) -> np.ndarray:
    '''
        Apply OpenCV's erode with ones fillter
    '''
        
    kernel = np.ones((kernel, ) * 2, np.uint8)
    return cv.erode(image, kernel, iterations = iteration)

def cleanup(image : np.ndarray) -> np.ndarray:
    '''
        Perform DroTrack image post processing

        image: array of pixel classification (np.ndarray)
    '''

    image = np.float32(image)
    image = dilate(image = image, kernel = __default_kernel, iteration = __default_iteration)
    image = erode(image = image, kernel = __default_kernel, iteration = __default_iteration)

    image = cv.medianBlur(np.uint8(image), __default_median_blur)

    return image

def image_bbox(image : np.ndarray, connectivity : int = 2, min_area : int = 0) -> np.ndarray:
    '''
        Find bounding box for blobs in segmentation

        Returns a list of RegionProperties of valid area
    '''

    labels = label(image, connectivity = connectivity)
    regions = regionprops(labels)

    return list(filter(lambda r: r.area >= min_area, regions))