
from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that run_infer_opts.py resides in

class mc_train_opts:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # TEST MCDataset

        self.parser.add_argument("--data_path",
                                 default="/home/roit/datasets/MC")
        self.parser.add_argument("--height", default=192)
        self.parser.add_argument("--width", default=256)
        self.parser.add_argument("--frame_idxs",default=[-1,0,1])
        self.parser.add_argument("--scales",default=[0,1,2,3])

        self.parser.add_argument("--batch_size",default=1)
        self.parser.add_argument("--num_workers",default=1)
        self.parser.add_argument("--mc",
                                 type=str,
                                 help="dataset to train on",
                                 # default="mc",
                                 default='kitti',
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "mc"])

        self.parser.add_argument("--splits",default='mc_lite')

    def args(self):
        self.options = self.parser.parse_args()
        return self.options
