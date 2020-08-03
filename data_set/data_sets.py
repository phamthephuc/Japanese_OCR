#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import glob
import os
from defination import ROOT_PATH
# ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

class ImageTextDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.listImagePaths, self.listLabels = self.load_sequence()
        self.nSamples = len(self.listLabels)
        print(self.nSamples)

    def load_sequence(self):
        sequence_folder = glob.glob(ROOT_PATH + "/" + self.root)
        # print(defination.ROOT_PATH)
        # print(sequence_folder)
        imagePathList = []
        labelList = []
        for sq in sequence_folder:
            list_images_file = glob.glob(os.path.join(sq, '*.jpg'))
            for filename in list_images_file:
                label = filename.split('/')[-1].split("_")[0]
                if (len(label) <= 23):
                    imagePathList.append(filename)
                    labelList.append(label)

        # print(imagePathList, labelList)
        return imagePathList, labelList

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        #print(index, len(self.listImagePaths))
        assert index < len(self.listImagePaths) , 'index range error'
        imgPath = self.listImagePaths[index]
        label = self.listLabels[index]
       # print(imgPath, label)
        try:
            img = Image.open(imgPath).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class paddingNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        imgW, imgH = img.size
        desireW, desireH = self.size

        ratio = desireH / imgH
        resizeW = int(ratio * imgW)
        size_resize = (resizeW, desireH)
        img = img.resize(size_resize, self.interpolation)

        padding = (0, 0 , desireW - resizeW, 0)
        img = ImageOps.expand(img, padding)
        img = self.toTensor(img)
        # print(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW

        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            # ratios_sort = ratios.copy().sort()
            # print("ratios", ratios)
            # print("max_ratio", ratios_sort)
            max_ratio = max(ratios)
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        # print("ratios", ratios)
        transform = paddingNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        # print("images", images[0].size(0))
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
