# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#从不同版本的gt中抽取， 然后压缩， 留来evaluation
from __future__ import absolute_import, division, print_function

import os
from path import Path
import argparse
import numpy as np
import PIL.Image as pil

from utils.official import readlines
from kitti_utils import generate_depth_map
from tqdm import tqdm
import matplotlib.pyplot as plt

def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')
    parser.add_argument('--split_dir',
                        type=str,
                        help='which split to export gt from',
                        default='/home/roit/datasets/splits/eigen',
                        choices=["eigen","eigen_zhou", "eigen_benchmark", "custom","custom_lite"])

    parser.add_argument('--data_path',#2012年版本最原始的
                        type=str,
                        help='path to the root of the KITTI data',
                        default='/home/roit/datasets/kitti')
    parser.add_argument('--depth_annotated_path',type=str,
                        default='/home/roit/bluep2/datasets/kitti_data_depth_annotated',
                        help='2015年又补充的')
    opt = parser.parse_args()

    split_dir = Path(opt.split_dir)
    lines = readlines(split_dir/ "test_files.txt")

    print("Exporting ground truth depths for {}".format(opt.split_dir))

    gt_depths = []

    data_path = Path(opt.data_path)#raw kitti path
    depth_annotated_path = Path(opt.depth_annotated_path)

    for line in tqdm(lines):

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if split_dir.stem == "eigen":#depth ground truth 在 场景文件夹中， 云图,full eigen split
            calib_dir = data_path/folder.split("/")[0]
            velo_filename = data_path/folder/"velodyne_points/data"/"{:010d}.bin".format(frame_id)
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        elif split_dir.stem == "eigen_benchmark":#后来补充的， ground-truth 在 ‘depth_annotated_path’,结果偏高
            gt_depth_path = depth_annotated_path/folder/"proj_depth"/"groundtruth"/"image_02"/"{:010d}.png".format(frame_id)
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256


        else:# opt.split =='custom':#根据png来取gt数据
            gt_depth_path = depth_annotated_path/folder/"proj_depth"/"groundtruth"/"image_02"/"{:010d}.png".format(frame_id)
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = split_dir/"gt_depths2.npz"

    print("Saving to {}".format(split_dir))

    np.savez_compressed(output_path, data=np.array(gt_depths))

def npz2img():
    gt  = "./splits/eigen/gt_depths2.npz"
    gt  = Path(gt)
    gt = np.load(gt,allow_pickle=True)

    gt = gt["data"]
    print(gt.shape)

if __name__ == "__main__":
    export_gt_depths_kitti()
    #npz2img()

