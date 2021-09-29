from os import sep
from PIL.Image import init
import numpy as np
import pandas as pd


class coordConverter(object):
    def __init__(self, homograph_path):
        homo_mat = pd.read_csv(homograph_path, header=None, sep = '\s+')
        self.homo_mat = np.array(homo_mat.iloc[:][:])
        self.homo_inv = np.array(np.matrix(self.homo_mat).I)
    def meter2pix(self, x, y):
        # Here the x is the width and y is the height in meter
        meter_coord = np.array([x,y,1]).reshape(3,1)
        result = np.matmul(self.homo_inv, meter_coord)
        return [int(result[0,0]/result[2,0]), int(result[1,0]/result[2,0])]

    def pix2meter(self, x, y):
        # Here the x is the width and y is the height in pixel
        pix_coord = np.array([x,y,1]).reshape(3,1)
        result = np.matmul(self.homo_mat, pix_coord)
        return [(result[0,0]/result[2,0]), (result[1,0]/result[2,0])]




# Use Seq_hotel as example.
# coord_path = "../dataset/ewap_dataset/seq_hotel/obsmat.txt"
# coord_real = pd.read_csv(coord_path, header=None, sep = '\s+')
# coord_real = np.array(coord_real.iloc[:][:])
# homo_path = "../dataset/ewap_dataset/seq_hotel/H.txt"


# import cv2
# from matplotlib import pyplot as plt

# vidcap = cv2.VideoCapture("../dataset/ewap_dataset/seq_hotel/seq_hotel.avi")
# ## Change the max_frame to test
# max_frame = 50
# count = 0
# index = 0
# converter = coordConverter(homo_path)

# while count < coord_real[max_frame, 0]:
#     s, image = vidcap.read()
#     count += 1
#     while coord_real[index, 0] < count :
#         plt.imshow(image)
#         x,y = converter.meter2pix(coord_real[index, 2], coord_real[index, 4])
#         plt.plot(y,x, "or", markersize=10)
#         index += 1
#     plt.show()




