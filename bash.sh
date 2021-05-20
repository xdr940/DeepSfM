#!/usr/bin/env bash

echo kitti
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti.yaml
clear

echo kitti1
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti1.yaml
clear


echo kitti2
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti2.yaml
clear

echo kitti3
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti3.yaml
clear

echo mc
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/mc.yaml
clear



