# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from networks.layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc#[64, 64, 128, 256, 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class DepthDecoder2(nn.Module):
    def __init__(self,
                 num_ch_enc,
                 scales=range(4),#scale 和mode 要設定一致
                 mode='train',
                 num_output_channels=1,
                 use_skips=True,
                 ):
        super(DepthDecoder2, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.mode = mode
        self.num_ch_enc = num_ch_enc#[64, 64, 128, 256, 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()

        self.convs[("upconv", 4, 0)] = ConvBlock(512, 256)
        self.convs[("upconv", 4, 1)] = ConvBlock(512, 256)

        self.convs[("upconv", 3, 0)] = ConvBlock(256, 128)
        self.convs[("upconv", 3, 1)] = ConvBlock(256, 128)

        self.convs[("upconv", 2, 0)] = ConvBlock(128, 64)
        self.convs[("upconv", 2, 1)] = ConvBlock(128, 64)

        self.convs[("upconv", 1, 0)] = ConvBlock(64, 32)
        self.convs[("upconv", 1, 1)] = ConvBlock(96, 32)

        self.convs[("upconv", 0, 0)] = ConvBlock(32, 16)
        self.convs[("upconv", 0, 1)] = ConvBlock(16, 16)



        if self.mode!='train':
            self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        else:
            for i in self.scales:
                self.convs[("dispconv", i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)


        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, *features):

        # f0 f1 f2 f3 f4
        self.outputs = {}
        ret = []

        x = features[4]

        # i = 4
        x = self.convs[("upconv", 4, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [features[3]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 4, 1)](x)
        #none

        # i=3
        x = self.convs[("upconv", 3, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [features[2]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)

        if self.mode=='train':
            ret.append(
                self.sigmoid(
                    self.convs[("dispconv", 3)](x)
                )
            )



        # i =2
        x = self.convs[("upconv", 2, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [features[1]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)

        if self.mode=='train':
            ret.insert(0,
                self.sigmoid(
                    self.convs[("dispconv", 2)](x)
                )
            )

        # i =1
        x = self.convs[("upconv", 1, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [features[0]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)

        if self.mode=='train':
            ret.insert(0,
                self.sigmoid(
                    self.convs[("dispconv", 1)](x)
                )
            )



        # i = 0
        x = self.convs[("upconv", 0, 0)](x)
        x = upsample(x)
        x = self.convs[("upconv", 0, 1)](x)

        if self.mode=='train':
            ret.insert(0,
                self.sigmoid(
                    self.convs[("dispconv", 0)](x)
                )
            )
            return ret
        else:
            return ret[0]

