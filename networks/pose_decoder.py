# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(in_channels=self.num_ch_enc[-1], out_channels=256, kernel_size=1)

        self.convs[("pose", 0)] = nn.Conv2d(in_channels=num_input_features * 256, out_channels=256,kernel_size= 3,stride= stride, padding=1)
        self.convs[("pose", 1)] = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=stride, padding=1)
        self.convs[("pose", 2)] = nn.Conv2d(in_channels=256, out_channels=6 * num_frames_to_predict_for, kernel_size=1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):

        last_features = [f[-1] for f in input_features]

        #cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]

        cat_features=[]
        for f in last_features:
            f = self.convs["squeeze"](f)
            fout = self.relu(f)
            cat_features.append(fout)

        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseDecoder2(nn.Module):
    def __init__(self,
                 num_ch_enc=[64,64,128,256,512],
                 out_num_poses=2,
                 stride=1
                 ):
        super(PoseDecoder2, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.out_num_poses = out_num_poses

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * out_num_poses, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, *input_features):

        last_features = input_features[-1]

        f = self.convs["squeeze"](last_features)
        fout = self.relu(f)


        x = self.convs[("pose", 0)](fout)
        x = self.relu(x)

        x = self.convs[("pose", 1)](x)
        x = self.relu(x)

        x = self.convs[("pose", 2)](x)


        out = x.mean(3).mean(2)

        poses = 0.01 * out.view(-1, self.out_num_poses, 1, 6)#3 frames, get [2,1,6]

        #axisangle = out[..., :3]
        #translation = out[..., 3:]

        return poses



def getPoseDecoder(mode):
    if mode=="fin-2out":
        return PoseDecoder2()
    else:
        pass

if __name__ == '__main__':
    pass