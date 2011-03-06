'''
Created on Feb 13, 2011

@author: akiva
'''
from cv import *
from numpy import fromstring
import numpy as np

from scipy import ndimage
from random import randint
import mahotas
import pymorph
import pylab

win_size = 20
pyramid = None
prev_pyramid = None

def array2cv(a):
    dtype2depth = {
          'uint8':   IPL_DEPTH_8U,
          'int8':    IPL_DEPTH_8S,
          'uint16':  IPL_DEPTH_16U,
          'int16':   IPL_DEPTH_16S,
          'int32':   IPL_DEPTH_32S,
          'float32': IPL_DEPTH_32F,
          'float64': IPL_DEPTH_64F,
      }
    try:  
        nChannels = a.shape[2]
    except:
        nChannels = 1
    cv_im = CreateImageHeader((a.shape[1], a.shape[0]),
            dtype2depth[str(a.dtype)],
            nChannels)
    SetData(cv_im, a.tostring(),
               a.dtype.itemsize * nChannels * a.shape[1])
    return cv_im

def cv2array(im): 
    depth2dtype = { 
          IPL_DEPTH_8U: 'uint8',
          IPL_DEPTH_8S: 'int8',
          IPL_DEPTH_16U: 'uint16',
          IPL_DEPTH_16S: 'int16',
          IPL_DEPTH_32S: 'int32',
          IPL_DEPTH_32F: 'float32',
          IPL_DEPTH_64F: 'float64',
      } 
    
    arrdtype = im.depth 
    a = fromstring(
           im.tostring(),
           dtype=depth2dtype[im.depth],
           count=im.width * im.height * im.nChannels) 
    a.shape = (im.height, im.width, im.nChannels) 
    return a 

def expand(array):
    if array == None:
        return None
    
    im = array * 10
    im = array2cv(im)
    
    #convert im to UINT8
    temp = CreateImage((im.width, im.height), 8, 1)
    ConvertScale(im, temp, 1 / 255.)
    
    # make the im bigger
    temp2 = CreateImage((im.width * 2, im.height * 2), 8, 1)
    Resize(temp, temp2, 1)
    temp3 = CreateImage((temp2.width, temp2.height),8,3)
    CvtColor(temp2,temp3, CV_GRAY2BGR)
#    print temp2
#    print temp2.width, temp2.height
    Smooth(temp2, temp2, CV_GAUSSIAN, 3,3)
    temp2 = thresh(temp2, 50)
    arr = cv2array(temp2)

#    arr[arr>0]=1
#    arr, num_labels = ndimage.label(arr)
#    arr = arr * 100

#    temp3 = array2cv(arr)
#    NamedWindow("temp3")
#    ShowImage("temp3",arr)
#    print temp2
#    print temp2.width, temp2.height
    return temp2, temp3

def getFeatures(im, MAX_COUNT, min_distance=10):
    # the default parameters
    quality = 0.001

    global win_size
    
    grey = im
    
    eig = CreateImage (GetSize (grey), 32, 1) 
    temp = CreateImage (GetSize (grey), 32, 1)
    # search the good points
    features = GoodFeaturesToTrack (
        grey, eig, temp,
        MAX_COUNT,
        quality, min_distance, None, 10
, 0, 0.04)

    # refine the corner locations
    features = FindCornerSubPix (
        grey,
        features,
        (win_size, win_size),  (-1, -1),
        (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03))
    return features

def drawFeatures(im, features):
    for the_point in features:
        Circle (im, (int(the_point[0]), int(the_point[1])), 3, (0,256,0,0), -1, 8, 0)
    return im


def getCenters(labels, nr_nuclei):
    centers = [(0,0) for i in xrange(1,nr_nuclei)]
    for i in xrange(1,nr_nuclei):
        non =  (labels==i).nonzero()
        centers[i-1] = (np.median(non[0]), np.median(non[1]))
    return centers

def getQuickFeatures(i):

    i = ndimage.gaussian_filter(i,2)
    t = mahotas.thresholding.otsu(i[i>0])
    i[i<t] = t
    i = ndimage.gaussian_filter(i,1)
    return getFeatures(i, 300);

def drawPoints(im, points):
    for the_point in points:
        curr = the_point
        prev = curr.prev
        color = curr.color
        while prev is not None:
            Line(im, (int(prev.x), int(prev.y)), (int(curr.x), int(curr.y)), color)
            curr = curr.prev
            prev = curr.prev
        Circle (im, (int(the_point.x), int(the_point.y)), the_point.radius, the_point.color, 1, 8, 0)
    return im

def thresh(image, min=20, max=255):
    thresh = CreateImage(GetSize(image), 8, 1)
    Threshold(image, thresh, min, max, CV_THRESH_TOZERO)
    return thresh

def calcOptimalFlow(prev_grey, grey, features, flags=0):
    global win_size
    global prev_pyramid
    global pyramid
    
    if not prev_pyramid:
        prev_pyramid = CreateImage(GetSize(grey), 8, 1)
        pyramid = CreateImage(GetSize(grey), 8, 1)
        
    features, status, track_error = CalcOpticalFlowPyrLK (
        prev_grey, grey, prev_pyramid, pyramid,
        features,
        (win_size, win_size), 3,
        (CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03),
        flags)
    prev_pyramid, pyramid = pyramid, prev_pyramid
    # set back the points we keep
    features = [ p for (st,p) in zip(status, features) if st]
    return features
