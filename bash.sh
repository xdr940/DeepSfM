#!/usr/bin/env bash
echo kitti
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti.yaml
clear

echo kitti1
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti1.yaml
clear

