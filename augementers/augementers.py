#!/usr/bin/python
# encoding: utf-8
from PIL import ImageFilter, ImageOps, Image
from skimage.util import random_noise
from skimage.filters import gaussian
import numpy as np
import PIL
import cv2

from augementers.warp_mls import WarpMLS

def toGrey(image):
    image = image.convert('L')
    return image

def invert(image):
    image = image.convert('L')
    image = ImageOps.invert(image)
    array = np.asarray(image)
    print(array)
    threshold = 100
    print(threshold)
    img_h, img_w = array.shape[:2]
    print(img_h, img_w)
    newArray = np.asarray([[array[i][j] if array[i][j] > threshold else threshold for j in range(img_w)] for i in range(img_h)], dtype=np.uint8)
    print(newArray)
    return Image.fromarray(newArray)
    #
    # image = image.convert('L')
    # return ImageOps.invert(image)

# def noise(image):
#     return random_noise(image, var=0.2**2)

def blur(image):
    image = image.convert('L')
    return image.filter(ImageFilter.GaussianBlur(0.5))


def distort(img, segment=10):
    src = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        np.random.randint(thresh) - half_thresh])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        img_h + np.random.randint(thresh) - half_thresh])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    im = Image.fromarray(dst).convert("L")
    return im


def stretch(img, segment=4):
    src = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    im = Image.fromarray(dst).convert("L")
    return im


def perspective(img):
    src = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    im = Image.fromarray(dst).convert("L")
    return im