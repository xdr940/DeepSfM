#!/usr/bin/env bash

echo fpv
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/fpv.yaml
clear

echo fpv1
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/fpv1.yaml
clear


echo fpv2
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/fpv2.yaml
clear

echo fpv3
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/fpv3.yaml
clear

echo fpv4
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/fpv4.yaml
clear

echo fpv5
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/fpv5.yaml
clear




