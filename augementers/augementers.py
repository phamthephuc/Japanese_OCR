#!/usr/bin/python
# encoding: utf-8
from PIL import Image, ImageOps
from skimage.util import random_noise
from skimage.filters import gaussian
import numpy as np

def invert(image):
    return ImageOps.invert(image)

def noisy(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def blur(image):
    return gaussian(image,sigma=1,multichannel=True)