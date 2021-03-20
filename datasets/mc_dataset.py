

from __future__ import absolute_import, division, print_function

import skimage.transform
import numpy as np
import PIL.Image as pil
from path import Path
import matplotlib.pyplot as plt

from datasets.mono_dataset_v2 import MonoDataset
import random
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
def relpath_split(relpath):
    relpath = relpath.split('/')
    traj_name=relpath[0]
    shader = relpath[1]
    frame = relpath[2]
    frame=frame.replace('.png','')
    return traj_name, shader, frame

class MCDataset(MonoDataset):
    def __init__(self,*args,**kwargs):
        super(MCDataset,self).__init__(*args,**kwargs)

        #self.full_res_shape = [1920,1080]#

        #
        # FOV = 35d
        # 960 = 1920/2
        # 960/fx = tan 35 =0.7-> fx = 1371
        #
        # 1920 * k[0] = 1371-> k0 = 0.714
        # 1080 * k[1 ]= 1371 -> k1 = 1.27
        # self.K=np.array([[0.714, 0, 0.5, 0],
        #                   [0, 1.27, 0.5, 0],
        #                   [0, 0, 1, 0],
        #                   [0, 0, 0, 1]], dtype=np.float32)




        #
        # #400/ fx = tan 35 =0.7 --> fx =571.428
        # #800 * k[0] = 571.428 ->> k0 = 0.714
        # #600* k1 = 571.428, k1 =0.952
        # self.full_res_shape = [800,600]#4:3
        # self.K = np.array([[0.714, 0, 0.5, 0],
        #                    [0, 0.952, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)


        #512/ fx = tan 35 =0.7 --> fx =731.219
        #1024 * k[0] = 731.219 ->> k0 = 1.4
        #768* k1 = 731.219, k1 =0.952

        self.full_res_shape = [1024, 768]#4:3
        self.K = np.array([[0.714, 0, 0.5, 0],
                           [0, 0.952, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)


        self.img_ext='.png'
        self.depth_ext = '.png'



    def check_depth(self):

        traj_name,shader,frame = relpath_split(self.filenames[0])

        depth_filename =Path(self.data_path)/traj_name/"depth"/"{:04d}.png".format(int(frame))

        return depth_filename.exists()

    def get_color(self, line, side, do_flip):
        path =self.__get_image_path__(line, side)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color



    def get_depth(self, line, side,  do_flip):
        path = self.__get_depth_path__(line, side)
        depth_gt = plt.imread(path)
        depth_gt = np.mean(depth_gt, axis=2)
        depth_gt = skimage.transform.resize(depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt*100#[0~1]


    def __get_image_path__(self, line, side):
        traj,shader,frame = relpath_split(line)
        reframe = "{:04d}".format(int(frame)+side)
        line = traj+'/'+shader+'/'+reframe+self.img_ext
        image_path = Path(self.data_path)/ line
        return image_path


    def __get_depth_path__(self, line, side):

        traj, shader, frame = relpath_split(line)
        reframe = "{:04d}".format(int(frame) + side)
        line = line.replace(frame, reframe)
        line = line.replace(shader,'depth')
        depth_path = Path(self.data_path) / line
        return depth_path



