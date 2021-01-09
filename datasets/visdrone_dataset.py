

from __future__ import absolute_import, division, print_function

import numpy as np
import PIL.Image as pil
from path import Path
from datasets.mono_dataset import MonoDataset


class VSDataset(MonoDataset):
    def __init__(self,*args,**kwargs):
        super(VSDataset,self).__init__(*args,**kwargs)

        self.full_res_shape = [1904,1071]#
        #self.full_res_shape = [800,600]#


        #FOV = 35d

        #960/fx = tan 35 =0.7-> fx = 1371

        # 1920 * k[0] = 1371-> k0 = 0.714
        # 1080 * k[1 ]= 1371 -> k1 = 1.27
        #self.K=np.array([[0.714, 0, 0.5, 0],
        #                   [0, 1.27, 0.5, 0],
        #                   [0, 0, 1, 0],
        #                   [0, 0, 0, 1]], dtype=np.float32)



        #952/fx = tan 35 =0.7-> fx = 1360

        # 1904 * k[0] = 1360-> k0 = 0.714
        # 1071 * k[1 ]= 1360 -> k1 = 1.28
        self.K = np.array([[0.714, 0, 0.5, 0],
                           [0, 1.27, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)


        self.img_ext='.jpg'
        #self.depth_ext = '.png'

        self.MaxDis = 255.
        self.MinDis = 0


    def check_depth(self):

       return False

    def get_color(self, folder, frame_index, side, do_flip):
        path =self.get_image_path(folder, frame_index)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


    def get_image_path(self, folder, frame_index):
        seq,frame = folder.split('/')
        frame = int(frame)

        f_str = "{:07d}{}".format(frame_index+frame, self.img_ext)
        image_path = Path(self.data_path)/ seq/"{}".format(f_str)
        return image_path



