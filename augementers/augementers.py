#!/usr/bin/python
# encoding: utf-8
from PIL import Image, ImageOps
from skimage.util import random_noise
from skimage.filters import gaussian

def invert(image):
    return ImageOps.invert(image)

def noise(image):
    return random_noise(image, var=0.2**2)

def blur(image):
    return gaussian(image,sigma=1,multichannel=True)