# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from my_trainer import Trainer
from opts.train_opts import train_opts
from opts.opts_yaml import YamlHandler
from opts.mc_train_opts import mc_train_opts



def main1():
    options = train_opts()
    opts = options.args()
    trainer = Trainer(opts)
    trainer.train()
    print('training over')
def main2():
    opts = YamlHandler('./opts/train_opts.yaml').read_yaml()
    trainer = Trainer(opts)
    trainer(opts)
    print('training over')

if __name__ == "__main__":
    main2()
