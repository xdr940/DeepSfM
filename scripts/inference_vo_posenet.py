# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
from torch.utils.data import DataLoader
from utils.official import readlines
from opts.md_vo_inferences_bian_opt import MD_vo_inferences_bian_opt
from datasets import CustomMonoDataset
from datasets import MCDataset
import networks
from tqdm import tqdm
from path import Path
from utils.inverse_warp import pose_vec2mat
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
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


    #assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
    #    "eval_split should be either odom_9 or odom_10"

    #sequence_id = int(opt.eval_split.split("_")[1])



    #filenames = readlines(
    #    os.path.join(os.path.dirname(__file__), "splits", "odom",
    #                 "test_files_{:02d}.txt".format(sequence_id)))
    # dataset = KITTIOdomDataset(opt.eval_pose_data_path, filenames, opt.height, opt.width,
    #                            [0, 1], 4, is_train=False)


    if opt.infer_file==None:
        filenames = readlines(Path('./splits')/opt.split/'test_files.txt')
    else:
        filenames = readlines(Path('./splits')/opt.split/opt.infer_file)
    if opt.split =="custom_mono":
        dataset =CustomMonoDataset(opt.dataset_path,
                                   filenames,
                                   opt.height,
                                   opt.width,
                                   [0, 1],
                                   1,
                                   is_train=False)
    elif opt.split =="mc":

        dataset = MCDataset(opt.dataset_path,
                                   filenames,
                                   opt.height,
                                   opt.width,
                                   [0, 1],
                                   1,
                                   is_train=False)


    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)



    #model

    weights_pose = torch.load(opt.posenet_path)
    pose_net = networks.PoseNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()






    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    print("-> eval "+opt.split)
    global_pose = np.identity(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]
    for inputs in tqdm(dataloader):
        for key, ipt in inputs.items():
            inputs[key] = ipt.cuda()

        pose = pose_net(inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)])#1,6
        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])  # 4X4
        global_pose = global_pose@ np.linalg.inv(pose_mat)


        poses.append(global_pose[0:3, :].reshape(1, 12))


    poses = np.concatenate(poses, axis=0)
    if opt.scale_factor:
            poses[:,3]*=opt.scale_factor#x-axis
            poses[:,11]*=opt.scale_factor#z-axis
    if opt.infer_file:
        dump_name = Path(opt.infer_file).stem +'.txt'
    else:
        dump_name = opt.dump_name
    np.savetxt(dump_name, poses, delimiter=' ', fmt='%1.8e')


def pose_format(options):
    pass
    poses = np.load(options.saved_npy)
    l,_,__ =poses.shape
    poses = poses.reshape(l,16)

    poses = poses[:,:12]
#    np.savetxt('poses.txt', poses, delimiter=' ', fmt='%1.8e')
    np.savetxt(options.eval_split+'.txt', poses, delimiter=' ', fmt='%1.8e')



if __name__ == "__main__":
    options = MD_vo_inferences_bian_opt().parse()

    main(options)
