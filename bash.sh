#fin

echo kitti.yaml
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti.yaml
clear


echo kitti2.yaml
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/kitti2.yaml
clear


echo mc.yaml
python scripts/train.py --settings /home/roit/aws/aprojects/DeepSfMLearner/opts/mc6.yaml
clear