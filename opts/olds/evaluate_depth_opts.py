
from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that run_infer_opts.py resides in

class evaluate_depth_opts:
    def __init__(self):
        # EVALUATION options
        self.parser = argparse.ArgumentParser(description="Monodepthv2 evaluation options")
        self.parser.add_argument('--root', type=str,
                                 default='/home/roit/aws/aprojects/xdr94_mono2')
        self.parser.add_argument("--data_path",
                                 type=str,
                                 #help="path to the training data",
                                 default='/home/roit/datasets/kitti/',
                                 #default = '/home/roit/datasets/MC/'
        )
        self.parser.add_argument("--depth_eval_path",
                                 help="",
                                 default='/home/roit/models/monodepth2_official/mono_640x192',#官方给出的模型文件夹
                                 #default="/home/roit/models/monodepth2/reproduction/models/weights_4",
                                 #default="/home/roit/models/monodepth2/pure_var_mask_median/models/weights_19",
                                 #default="/home/roit/models/monodepth2/avg_loss/models/weights_19",
                                 #default="/home/roit/models/monodepth2/identicalmask/models/weights_4",
                                 #default="/home/roit/models/monodepth2/identical+var/models/weights_7",
                                 #default="/home/roit/models/monodepth2/identical_var_mean/last_model",
                                 #default="/home/roit/models/monodepth2/fullwitherodil/last_model",
                                 #default="/home/roit/models/monodepth2/fullwitherodil10epoch/last_model",
                                 #default="/home/roit/models/monodepth2/05-14-13:57/models/weights_4"#
                                 #default="/home/roit/models/monodepth2/checkpoints/05-14-19:27/models/weights_19"#
                                 #default="/home/roit/models/monodepth2/05-27-17:48/models/weights_10"
                                 #default = "/home/roit/models/monodepth2/checkpoints/05-15-14:40/models/weights_19"
                                 # default = "/home/roit/models/monodepth2/checkpoints/05-15-14:40/models/weights_19"
                                 #default="/home/roit/models/monodepth2/checkpoints/05-18-00:34/models/weights_4"
                                 #default="/home/roit/models/monodepth2/checkpoints/05-20-00:06/models/weights_19"
                                 #default = "/home/roit/models/monodepth2/checkpoints/06-02-06:59/models/weights_19"
                                 #default = "/home/roit/models/monodepth2/checkpoints/06-15-22:30/models/weights_24"
                                 #default='/home/roit/models/monodepth2_official/mono_640x192',
                                 # default='/home/roit/bluep2/models/monodepth2/05231920/models/weights_4'
                                 #default='/media/roit/hard_disk_2/Models/monodepth2_train/06152230/models/weights_14'

                                 )
        self.parser.add_argument('--test_files',default='test_files.txt')
        self.parser.add_argument("--test_dir",
                                 type=str,
                                 default="/home/roit/datasets/splits/eigen",  # eigen
                                 #choices=["eigen_zhou","eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "custom", "mc","mc_lite"],
                                 help="which split to run eval on")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 default=True)
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)


        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
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


        self.parser.add_argument("--eval_batch_size", default=16, type=int)


        self.parser.add_argument("--min_depth",type=float,help="minimum depth",default=0.1)#这里度量就代表m
        self.parser.add_argument("--max_depth",type=float,help="maximum depth",default=80.0)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
