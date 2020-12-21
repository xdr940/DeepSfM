

from __future__ import absolute_import, division, print_function

import os
from path import Path
import argparse
import numpy as np
import PIL.Image as pil

from utils.official import readlines
from kitti_utils import generate_depth_map
from tqdm import tqdm

parser = argparse.ArgumentParser(description='export_gt_depth')
parser.add_argument('--split',
                    type=str,
                    help='which split to export gt from',
                    default='mc',
                    choices=["mc", "custom", "mc_lite"])

parser.add_argument('--data_path',
                    type=str,
                    help='path to the root of the mc data',
                    default='/home/roit/datasets/MC')

opt = parser.parse_args()


def export_gt_depths_mc(opt):


    split_folder = Path('.') / "splits" / opt.split
    lines = readlines(split_folder / "test_files.txt")

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []

    data_path = Path(opt.data_path)  # raw kitti path

    for line in tqdm(lines):

        block,trajectory,data_type, frame_id = line.split('/')
        frame_id = int(frame_id)



        if opt.split == "mc" or "mc_lite":  # 后来补充的， ground-truth 在 ‘depth_annotated_path’,结果偏高
            gt_depth_path = data_path / block / trajectory/"depth" / "{:04d}.png".format(frame_id)
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32)[:,:,0]# / 255

            gt_depths.append(gt_depth.astype(np.float32))
        else:
            print('no data set selected')
            return
    output_path = split_folder / "gt_depths.npz"

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))

if __name__ == "__main__":
    export_gt_depths_mc(opt)