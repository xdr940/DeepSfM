# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn




class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        return out
        #axisangle = out[..., :3]
        #translation = out[..., 3:]

        #return axisangle, translation



class PoseC3D(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseC3D, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.conv3d = nn.Conv3d(3, 16, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3))
        self.bn3d = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool3d = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        #self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn3d(x)
        x = self.relu(x)
        x = self.maxpool3d(x)

        x = torch.squeeze(x, dim=2)


        for i in range(1,self.num_convs):#[8,64,320,96]
            x = self.convs[i](x)
            x = self.relu(x)

        x = self.pose_conv(x)
        out = x.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        return out


def getPoseNet(mode):
    if mode=="3in":
        return PoseCNN(3)
    elif mode == "3din":
        return PoseC3D(3)
    elif mode =='2in':
        return PoseCNN(2)


if __name__ == '__main__':
    network = getPoseNet("3d-in")
    example_inputs = torch.rand(8, 3,3, 640, 192)
    out = network(example_inputs)
    print(out.shape)
    #
    encoder_out = torch.onnx.export(model=network,
                                    args=example_inputs,
                                    input_names=["input"],
                                    f= "./pose_cnn_3in.onnx",
                                    #    output_names=['f0', 'f1', 'f2', 'f3', 'f4'],
                                    verbose=True,
                                    export_params=True  # 带参数输出
                                    )
