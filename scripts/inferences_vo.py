# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from networks.layers import transformation_from_parameters
from utils.official import readlines
from opts.md_vo_inferences import MD_vo_inferences
from datasets import CustomMonoDataset
import networks
from tqdm import tqdm
from path import Path

#run from txt


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse

@torch.no_grad()
def main(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    #assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
    #    "eval_split should be either odom_9 or odom_10"

    #sequence_id = int(opt.eval_split.split("_")[1])



    #filenames = readlines(
    #    os.path.join(os.path.dirname(__file__), "splits", "odom",
    #                 "test_files_{:02d}.txt".format(sequence_id)))
    # dataset = KITTIOdomDataset(opt.eval_pose_data_path, filenames, opt.height, opt.width,
    #                            [0, 1], 4, is_train=False)



    filenames = readlines(Path('./splits')/opt.split/'test_files.txt')

    dataset =CustomMonoDataset(opt.dataset_path,
                               filenames,
                               opt.height,
                               opt.width,
                               [0, 1],
                               1,
                               is_train=False)



    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)



    #model
    pose_encoder_path = Path(opt.load_weights_folder)/"pose_encoder.pth"
    pose_decoder_path = Path(opt.load_weights_folder)/ "pose.pth"

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    print("-> eval "+opt.split)
    for inputs in tqdm(dataloader):
        for key, ipt in inputs.items():
            inputs[key] = ipt.cuda()

        all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

        features = [pose_encoder(all_color_aug)]
        axisangle, translation = pose_decoder(features)


        pred_pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
        pred_pose = pred_pose.cpu().numpy()
        pred_poses.append(pred_pose)

    pred_poses = np.concatenate(pred_poses)
    length =    pred_poses.shape[0]
    pred_poses.resize([length,16])
    pred_poses = pred_poses[:,:12]
    filename = opt.dump_name
    np.savetxt(filename, pred_poses, delimiter=' ', fmt='%1.8e')

    print("-> Predictions saved to", filename)


def pose_format(options):
    pass
    poses = np.load(options.saved_npy)
    l,_,__ =poses.shape
    poses = poses.reshape(l,16)

    poses = poses[:,:12]
#    np.savetxt('poses.txt', poses, delimiter=' ', fmt='%1.8e')
    np.savetxt(options.eval_split+'.txt', poses, delimiter=' ', fmt='%1.8e')



if __name__ == "__main__":
    options = MD_vo_inferences().parse()
    main(options)
