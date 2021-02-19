
from datasets.kitti_dataset import KITTIRAWDataset
from torch.utils.data import DataLoader

from utils.official import readlines

if __name__ == '__main__':

    train_filenames = readlines('/home/roit/datasets/splits/eigen_zhou/val_files.txt')
#     train_filenames = readlines('/home/roit/datasets/splits/eigen_std/test_files.txt')


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
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )

    for data in train_loader:
        pass