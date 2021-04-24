

from __future__ import absolute_import, division, print_function

import skimage.transform
import numpy as np
import PIL.Image as pil
from path import Path
import matplotlib.pyplot as plt
import os

import random
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from kitti_utils import generate_depth_map

from datasets.mono_dataset_v2 import MonoDataset


def relpath_split(relpath):
    relpath = relpath.split('/')
    date=relpath[0]
    scene = relpath[1]
    camera = relpath[2]
    #data
    frame = relpath[4]
    frame = frame.replace('.png', '')
    return date,scene,camera,frame

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
#
class KITTIRAWDataset(MonoDataset):
    def __init__(self,*args,**kwargs):
        super(KITTIRAWDataset,self).__init__(*args,**kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.img_ext = '.png'#kitti default
        self.full_res_shape = [1242, 375]
        self.camera_num_map = {"2": 2, "3": 3, "image_02": 2, "image_03": 3}




    def check_depth(self):

        date,scene,sensor,frame = relpath_split(self.filenames[0])
        sensor = 'velodyne_points'
        velo_filename = os.path.join(
            date,
            scene,
            sensor,
            'data',
            "{:010d}.bin".format(int(frame)))


        depth_filename =Path(self.data_path)/velo_filename

        return depth_filename.exists()

    def get_color(self, line, side, do_flip):
        path =self.__get_image_path__(line, side)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color



    def get_depth(self, line, side,  do_flip):

        date, scene, sensor, frame = relpath_split(line)
        reframe = str(int(frame) + side)

        velo_filename = os.path.join(
            date,
            scene,
            "velodyne_points",
            'data',
            "{:010d}.bin".format(int(reframe)))

        depth_path = Path(self.data_path) / velo_filename

        calib_path = os.path.join(
            self.data_path,
            date
        )

        depth_gt = generate_depth_map(calib_path, depth_path, self.camera_num_map[sensor])

        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


    def __get_image_path__(self, line, side):
        date, scene, sensor, frame = relpath_split(line)



        reframe = "{:010d}".format(int(frame)+side)
        path = os.path.join(
            date,
            scene,
            sensor,
            'data',
            reframe
        )
        image_path = Path(self.data_path)/ path+'.png'

        # #3帧图像测试需要, 主要是针对eigen split那些official eval
        # if not Path.exists(image_path) and int(frame)+side < 0:
        #     path = os.path.join(
        #         date,
        #         scene,
        #         sensor,
        #         'data',
        #         "{:010d}".format(0)
        #     )
        # elif not Path.exists(image_path) and int(frame)+side > 0:
        #     path = os.path.join(
        #         date,
        #         scene,
        #         sensor,
        #         'data',
        #         "{:010d}".format(int(frame))
        #     )
        #
        # image_path = Path(self.data_path) / path + '.png'
        return image_path


    def __get_depth_path__(self, line, side):
        date, scene, sensor, frame = relpath_split(self.filenames[0])
        reframe = str(int(frame)+side)

        velo_filename = os.path.join(
            date,
            scene,
            "velodyne_points",
            'data',
            "{:010d}.bin".format(int(reframe)))

        depth_path = Path(self.data_path) / line
        return depth_path



