# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from my_trainer import Trainer
from utils.yaml_wrapper import YamlHandler


def main():
    opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/mc.yaml').read_yaml()
    # opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/kitti.yaml').read_yaml()

    trainer = Trainer(opts)
    trainer(opts)
    print('training over')

if __name__ == "__main__":
     main()
