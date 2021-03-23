

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
# class MonoDataset(data.Dataset):
#     """Superclass for monocular dataloaders
#
#     Args:
#         data_path
#         filenames
#         height
#         width
#         frame_idxs
#         num_scales
#         is_train
#         img_ext
#     """
#     def __init__(self,
#                  data_path,
#                  filenames,
#                  height,
#                  width,
#                  frame_sides,
#                  num_scales,
#                  is_train=False,
#                  img_ext='.png'):
#         super(MonoDataset, self).__init__()
#
#         self.data_path = data_path
#         self.filenames = filenames#list , like '2011_09_26/2011_09_26_drive_0001_sync 1 l'
#         self.height = height
#         self.width = width
#         self.num_scales = num_scales
#         self.interp = Image.ANTIALIAS
#
#         self.frame_sides = frame_sides
#
#         self.is_train = is_train#unsuper train or evaluation
#         self.img_ext = img_ext
#
#         self.loader = pil_loader
#         self.to_tensor = transforms.ToTensor()
#
#         # We need to specify augmentations differently in newer versions of torchvision.
#         # We first try the newer tuple version; if this fails we fall back to scalars
#         try:
#             self.brightness = (0.8, 1.2)
#             self.contrast = (0.8, 1.2)
#             self.saturation = (0.8, 1.2)
#             self.hue = (-0.1, 0.1)
#             transforms.ColorJitter.get_params(
#                 self.brightness,
#                 self.contrast,
#                 self.saturation,
#                 self.hue
#             )
#         except TypeError:
#             self.brightness = 0.2
#             self.contrast = 0.2
#             self.saturation = 0.2
#             self.hue = 0.1
#
#         self.resizor = {}
#         for i in range(self.num_scales):
#             s = 2 ** i
#             self.resizor[i] = transforms.Resize((self.height // s, self.width // s),
#                                                interpolation=self.interp)
#
#         self.load_depth = self.check_depth()
#         #self.load_depth = False
#
#
#
#     #private
#     def preprocess(self, inputs, color_aug):
#         """Resize colour images to the required scales and augment if required
#
#         We create the color_aug object in advance and apply the same augmentation to all
#         images in this item. This ensures that all images input to the pose network receive the
#         same augmentation.
#         """
#         for k in list(inputs):
#             if "color" in k:
#                 type, side, _ = k
#                 for scale in range(self.num_scales):
#                     inputs[(type, side, scale)] = self.resizor[scale](inputs[(type, side, scale - 1)])
#
#         for k in list(inputs):
#             f = inputs[k]
#             if "color" in k:
#                 type, side, scale= k
#                 inputs[(type, side, scale)] = self.to_tensor(f)
#                 if scale<=0:
#                     inputs[(type + "_aug", side, scale)] = self.to_tensor(color_aug(f))
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, index):
#         """Returns a single training item(inputs ) from the datasets as a dictionary.
#
#         Values correspond to torch tensors.
#         Keys in the dictionary are either strings or tuples:
#
#             ("color", <frame_id>, <scale>)          for raw colour images,
#             ("color_aug", <frame_id>, <scale>)      for augmented colour images,
#             ("K", scale) or ("inv_K", scale)        for camera intrinsics,
#             "stereo_T"                              for camera extrinsics, and
#             "depth_gt"                              for ground truth depth maps.
#
#         <frame_id> is either:
#             an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
#         or
#             "s" for the opposite image in the stereo pair.
#
#         <scale> is an integer representing the scale of the image relative to the fullsize image:
#             -1      images at native resolution as loaded from disk#1242x362
#             0       images resized to (self.width,      self.height     )#640x192
#             1       images resized to (self.width // 2, self.height // 2)
#             2       images resized to (self.width // 4, self.height // 4)
#             3       images resized to (self.width // 8, self.height // 8)
#         """
#         inputs = {}
#
#         do_color_aug = self.is_train and random.random() > 0.5
#         do_flip = self.is_train and random.random() > 0.5
#         do_rotation = self.is_train and random.random()>0.5
#
#         split_line = self.filenames[index]
#
#
#         #img sides
#         for side in self.frame_sides:
#             try:
#                 if "0021" in split_line:
#                     pass
#                 inputs[("color", side, -1)] = self.get_color(split_line, side,  do_flip)  # inputs得到scale == -1的前 中后三帧
#             except:
#                 import os
#                 os.system('clear')
#                 print(split_line)
#                 exit(-1)
#
#         # adjusting intrinsics to match each scale in the pyramid
#         for scale in range(self.num_scales):
#             K = self.K.copy()
#
#             K[0, :] *= self.width // (2 ** scale)
#             K[1, :] *= self.height // (2 ** scale)
#
#             inv_K = np.linalg.pinv(K)
#
#             inputs[("K", scale)] = torch.from_numpy(K)
#             inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
#
#
#
#         if do_color_aug:
#             color_aug = transforms.ColorJitter.get_params(#对图像进行稍微处理，aug 但是要保证深度一致
#                 self.brightness, self.contrast, self.saturation, self.hue)
#         else:
#             color_aug = (lambda x: x)
#
#         self.preprocess(inputs, color_aug)#scalse,aug generate to 38
#
#
#
#         for i in self.frame_sides:
#             del inputs[("color", i, -1)]#删除原分辨率图
#             del inputs[("color_aug", i, -1)]#删除原分辨率曾广图
#
#
#
#         if self.load_depth:
#             depth_gt = self.get_depth(split_line, 0,  do_flip)
#             inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
#             inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
#
#
#
#
#
#         return inputs
#
#
#     #多态, 继承类用
#     def get_color(self, line, side, do_flip):
#         raise NotImplementedError
#
#     def check_depth(self):
#         raise NotImplementedError
#
#     def get_depth(self, line, side, do_flip):
#         raise NotImplementedError

class KITTIRAWDataset(MonoDataset):
    def __init__(self,*args,**kwargs):
        super(KITTIRAWDataset,self).__init__(*args,**kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)


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



