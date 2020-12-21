
from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that run_infer_opts.py resides in


class MD_vo_inferences:
    def __init__(self):
        # EVALUATION options
        self.parser = argparse.ArgumentParser(description="Monodepthv2 evaluation options")
        self.parser.add_argument('--root', type=str,
                                 default='/home/roit/aws/aprojects/xdr94_mono2')

        self.parser.add_argument("--load_weights_folder",
                                 help="",
                                 default="/home/roit/models/monodepth2_official/mono_640x192"

                                 )

        self.parser.add_argument("--dataset_path",
                                 default='/home/roit/datasets/Binjiang',
                                 help='must a success sequence')
        self.parser.add_argument("--dump_name",default='infer_vo_poses.txt')

        self.parser.add_argument("--height", default=192)
        self.parser.add_argument("--width", default=640)
        self.parser.add_argument("--split",
                                 type=str,
                                 default="custom_mono",  # eigen
                                 choices=["custom_mono"],
                                 help="which split to run eval on")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)

        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir", default='eval_out_dir',
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",  # ??
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")


        self.parser.add_argument("--eval_pose_save_path", default="./")
        self.parser.add_argument("--eval_batch_size", default=8, type=int)
        self.parser.add_argument("--batch_size", default=1, type=int)

        self.parser.add_argument("--min_depth", type=float, help="minimum depth", default=0.1)  # 这里度量就代表m
        self.parser.add_argument("--max_depth", type=float, help="maximum depth", default=80.0)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)
        self.parser.add_argument("--pose_format", default=True)
        self.parser.add_argument("--saved_npy", default="odom_04.npy")
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
