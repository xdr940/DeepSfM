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




class run_inference_opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Simple testing funtion for Monodepthv2 models.')

        self.parser.add_argument('--image_path', type=str,
                            default='/home/roit/datasets/npmcm2020/ew',
                            help='path to a test image or folder of images')
        self.parser.add_argument('--out_path', type=str, default=None, help='path to a test image or folder of images')
        self.parser.add_argument('--npy_out', default=False)
        self.parser.add_argument('--model_name', type=str,
                            help='name of a pretrained model to use',
                            default='mono_640x192',
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
        self.parser.add_argument('--model_path',
                            type=str,
                            #default='/home/roit/models/monodepth2_official',
                            # default='/home/roit/models/monodepth2/identical_var_mean',
                            default='/home/roit/models/monodepth2_official/',

                                 help='root path of models')
        self.parser.add_argument('--ext', type=str, help='image extension to search for in folder'
                            # default="*.jpg"
                            )
        self.parser.add_argument("--no_cuda", help='if set, disables CUDA', action='store_true')
        self.parser.add_argument("--out_ext", default="*.png")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options