# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo



class Res3DEncoder0(models.ResNet):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, block, layers,num_input_images=1):
        super(Res3DEncoder0, self).__init__(block, layers)




        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)






    def forward(self, input_image):
        self.features = []
       # x = (input_image - 0.45) / 0.225
        x = self.conv1(input_image)
        x = self.bn1(x)
        x = self.relu(x)
        self.features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.features.append(x)
        x = self.layer2(x)
        self.features.append(x)
        x = self.layer3(x)
        self.features.append(x)
        x = self.layer4(x)
        self.features.append(x)

        return self.features



class Res3DEncoder1(models.ResNet):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, block, layers,num_input_images=3):
        super(Res3DEncoder1, self).__init__(block, layers)




        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)






    def forward(self, input_image):
        self.features = []
       # x = (input_image - 0.45) / 0.225
        x = self.conv1(input_image)
        x = self.bn1(x)
        x = self.relu(x)
        self.features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.features.append(x)
        x = self.layer2(x)
        self.features.append(x)
        x = self.layer3(x)
        self.features.append(x)
        x = self.layer4(x)
        self.features.append(x)

        return self.features



class Res3DEncoder2(models.ResNet):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, block, layers):
        super(Res3DEncoder2, self).__init__(block, layers)




        self.inplanes = 64
        self.conv3d = nn.Conv3d(3, 64, kernel_size=(3, 7,7),stride=(1,1,1), padding=(1, 3, 3))
        self.bn3d = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool3d = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)






    def forward(self, input_image):
        self.features = []
       # x = (input_image - 0.45) / 0.225
        x = self.conv3d(input_image)
        x = self.bn3d(x)
        x = self.relu(x)
        x = self.maxpool3d(x)

        x = torch.squeeze(x,dim=2)
        self.features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.features.append(x)
        x = self.layer2(x)
        self.features.append(x)
        x = self.layer3(x)
        self.features.append(x)
        x = self.layer4(x)
        self.features.append(x)

        return self.features

def getEncoder(model_mode):
    if model_mode==1:
        return Res3DEncoder1(layers=[2,2,2,2],
                         block=models.resnet.BasicBlock,
                             num_input_images=3)
    elif model_mode==2:
        return Res3DEncoder2(layers=[2, 2, 2, 2],
                             block=models.resnet.BasicBlock,
                             num_input_images=3)



if __name__ == '__main__':


    # encoder= getEncoder(1)
    #
    # example_inputs= torch.rand(8, 9, 640, 192)

    encoder = getEncoder(2)
    example_inputs = torch.rand(8, 3, 3, 640, 192)



    features = encoder(example_inputs)




    #
    # out = encoder3d(example_inputs)
    #
    #
    encoder_out = torch.onnx.export(model=encoder,
                                    args=example_inputs,
                                    input_names=["input"],
                                    f= "./res3d.onnx",
                                    #    output_names=['f0', 'f1', 'f2', 'f3', 'f4'],
                                    verbose=True,
                                    export_params=True  # 带参数输出
                                    )
    pass