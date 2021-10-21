
from datasets.kitti_dataset_v2 import KITTIRAWDataset
from datasets.mc_dataset import MCDataset
from torch.utils.data import DataLoader

from utils.official import readlines
import matplotlib.pyplot as plt

def kittiv2_test():
    train_filenames = readlines('/home/roit/datasets/splits/eigen_zhou_std/train_files.txt')

    train_dataset = KITTIRAWDataset(  # KITTIRAWData
        data_path='/home/roit/datasets/kitti',
        filenames=train_filenames,
        height=192,
        width=640,
        frame_sides=[-1, 0, 1],  # kitti[0,-1,1],mc[-1,0,1]
        num_scales=4,
        is_train=True,
        img_ext='.png'
    )

    train_loader = DataLoader(  # train_datasets:KITTIRAWDataset
        dataset=train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    for data in train_loader:
        print('ok')
        print('okk')
def mc_test():
    pass

if __name__ == '__main__':
    kittiv2_test()
    # train_filenames = readlines('/home/roit/datasets/splits/eigen_zhou/val_files.txt')
