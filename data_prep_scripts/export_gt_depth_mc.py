

from __future__ import absolute_import, division, print_function

from path import Path
import argparse
import numpy as np
import PIL.Image as pil
from utils.official import np_normalize_image
from utils.official import readlines
from tqdm import tqdm
from datasets.mc_dataset import relpath_split
parser = argparse.ArgumentParser(description='export_gt_depth')
parser.add_argument('--split',
                    type=str,
                    help='which split to export gt from',
                    default='/home/roit/datasets/splits/mc/mcv3-sildurs-2k-12345')
parser.add_argument('--base',default='test.txt')
parser.add_argument('--data_path',
                    type=str,
                    help='path to the root of the mc data',
                    default='/home/roit/datasets/mcv3')
parser.add_argument('--depth_range',default=[1e-3,576])



opt = parser.parse_args()


def export_gt_depths_mc(opt):


    split_folder = Path(opt.split)
    lines = readlines(split_folder / opt.base)

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []

    data_path = Path(opt.data_path)  # raw kitti path

    for line in tqdm(lines):

        traj_name, shader, frame = relpath_split(line)



        if opt.split == "mc" or "mc_lite":  # 后来补充的， ground-truth 在 ‘depth_annotated_path’,结果偏高

            pass
            gt_depth_path = data_path / traj_name / "depth" / "{:04d}.png".format(int(frame))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32)
            gt_depth = gt_depth.sum(axis=2)
            gt_depth =np_normalize_image(gt_depth)
            gt_depths.append(gt_depth.astype(np.float32))
        else:
            print('no data set selected')
            return
    output_path = split_folder / "gt_depths.npz"

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))

if __name__ == "__main__":
    export_gt_depths_mc(opt)