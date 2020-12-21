
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that run_infer_opts.py resides in


class run_from_txt_opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Simple testing funtion for Monodepthv2 models.')



       #-------------------------------
        self.parser.add_argument('--out_path', type=str, default='./08171315_uav0000239_11136_s',
                            help='path to a test image or folder of images')
        self.parser.add_argument('--model_name', type=str,
                            help='name of a pretrained model to use',
                            default='weights_9',
                                 choices=[
                                "last_model",
                                "mono_640x192",
                                "stereo_640x192",
                                "mono+stereo_640x192",
                                "mono_no_pt_640x192",
                                "stereo_no_pt_640x192",
                                "mono+stereo_no_pt_640x192",
                                "mono_1024x320",
                                "stereo_1024x320",
                                "mono+stereo_1024x320"])
        self.parser.add_argument('--model_path', type=str,
                                 default='/home/roit/models/monodepth2/vsd/08171031/models',#baseline
                                 #default='/home/roit/models/monodepth2/fullwitherodil',
                                 #default='/home/roit/models/monodepth2/identical_var_mean',
                                 #default='/home/roit/models/monodepth2/visdrone/06-06-14:43/models',
                                 #default='/home/roit/models/monodepth2/custom_mono/07-12-13:41/models',
                                 #default='/media/roit/970evo/home/roit/models/monodepth2/MC/06021108/models',


                                 help='root path of models')
        self.parser.add_argument('--ext', type=str, help='image extension to search for in folder', default=".png")
        self.parser.add_argument("--no_cuda", help='if set, disables CUDA', action='store_true')
        self.parser.add_argument("--out_ext", default="*.png")

        #----------------------
        self.parser.add_argument('--txt_files',default='uav0000239_11136_s.txt')

        self.parser.add_argument("--wk_root",default="/home/roit/aws/aprojects/xdr94_mono2")

        self.parser.add_argument("--frame_ids",default=[-1,0,1])
        self.parser.add_argument("--results",
                                 default=["depth",
                                          #"var_mask",
                                          #"mean_mask",
                                          #"identical_mask",
                                          #"final_mask"
                                          ])
        self.parser.add_argument('--dataset_path', type=str,
                                 #default='/home/roit/datasets/MC',
                                 #default='/home/roit/datasets/Binjiang',
                                 #default='/970evo/home/roit/datasets/kitti',
                                 #default='/bluep2/datasets/VisDrone2',
                                 default="/home/roit/datasets/VSD",
                                 help='path to a test image or folder of images')
        self.parser.add_argument("--split",
                                 default="visdrone",
                                          # "custom",
                                          # "custom_lite"
                                          # "custom_mono",
                                          # "eigen",
                                          # 'eigen_zhou',
                                          # 'eigen_dali'
                                          # "mc",
                                          # "mc_lite",
                                          # "visdrone",
                                          # "visdrone_lite"
                                 )

        self.parser.add_argument("--as_name_sort", default=True)
        self.parser.add_argument("--out_full_shape", default=True)
        self.parser.add_argument('--npy_out', default=False)


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
