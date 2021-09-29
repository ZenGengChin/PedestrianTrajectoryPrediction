import os
from coordinate import coordConverter
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
import cv2
from matplotlib import pyplot as plt
from torch import Tensor
import torch

def LoadImage(Frame, videopath):
    '''Frame list or array of int'''
    count = 1
    images = [0 for i in range(len(Frame))]
    vidcap = cv2.VideoCapture(videopath)
    s, imageread = vidcap.read()
    while s:
        for indx, frame in enumerate(Frame):
            if frame == count:
                images[indx] = imageread
        s, imageread = vidcap.read()
        count += 1
    return Tensor(images).float().permute(0, 3, 1, 2)

