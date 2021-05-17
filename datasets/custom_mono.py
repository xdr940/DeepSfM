

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
from path import Path
from datasets.mono_dataset_v2 import MonoDataset

from torch.utils.data import DataLoader




class CustomMonoDataset(MonoDataset):
    def __init__(self,*args,**kwargs):
        super(CustomMonoDataset,self).__init__(*args,**kwargs)

        #self.full_res_shape = [1904,1071]#
        #self.full_res_shape = [800,600]#


        #FOV = 35d

        #960/fx = tan 35 =0.7-> fx = 1371

         #1920 * k[0] = 1371-> k0 = 0.714
        # 1080 * k[1 ]= 1371 -> k1 = 1.27
        self.K=np.array([[0.714, 0, 0.5, 0],
                           [0, 1.27, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)



        #952/fx = tan 35 =0.7-> fx = 1360

        # 1904 * k[0] = 1360-> k0 = 0.714
        # 1071 * k[1 ]= 1360 -> k1 = 1.28
        #self.K = np.array([[0.714, 0, 0.5, 0],
        #                   [0, 1.27, 0.5, 0],
        #                   [0, 0, 1, 0],
        #                   [0, 0, 0, 1]], dtype=np.float32)

        self.img_ext = '.png'
        self.scene_reg = "{:04d}"
        self.frame_reg = "{:04d}"

    def relpath_split(self,relpath):
        relpath = relpath.split('/')
        scene = relpath[0]
        frame = relpath[1]
        frame = frame.replace(self.img_ext, '')
        return scene, frame

    def __get_image_path__(self, line, side):
        scene,frame = self.relpath_split(line)
        reframe = (self.frame_reg).format(int(frame)+side)
        line = scene+'/'+reframe+self.img_ext
        image_path = Path(self.data_path)/ line
        return image_path



    def get_color(self, line, side, do_flip):
        path = self.__get_image_path__(line, side)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):

        return False





class CustomMonoVoDataset(MonoDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(MonoDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:04d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


if __name__ == '__main__':
    dataset = CustomMonoDataset(
        data_path='/home/roit/datasets/VSD',
        filenames=['uav0000077_00000_s/0000005.jpg','uav0000077_00000_s/0000022.jpg','uav0000077_00000_s/0000033.jpg'],
        height=320,
        width=384,
        frame_sides = [-1,0,1],
        num_scales=4,
        mode="train",
        img_ext='.jpg'
    )
    dataloader =DataLoader(dataset)

    for item in dataloader:
        pass