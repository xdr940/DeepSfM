#move files
from path import Path
import re
from utils.official import readlines
from utils.img_process import COLORMAPS
from matplotlib import cm

import os
from tqdm import  tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
def extract_MC():
    cp_img=False
    cp_gt =True
    dataset = Path("/home/roit/datasets/mcv5")
    wk_root = Path('/home/roit/aws/aprojects/xdr94_mono2')
    lines = '/home/roit/datasets/splits/mc/mcv5-sildurs-e-10k-12345-s/test.txt'
    dump_base = Path('/home/roit/bluep2/test_out/mc/apr/mcv5-sildurs-e-10k-12345-s')
    shader = 'sildurs-e'

    save_cmap='magma'


    dump_base.mkdir_p()







    img_dump = wk_root/'color'
    img_dump.mkdir_p()

    gt_dump = wk_root/'test_files'
    gt_dump.mkdir_p()
    (dump_base / 'img').mkdir_p()
    (dump_base / 'depth').mkdir_p()


    files = readlines(lines)

    for item in tqdm(files):
        if cp_img:
            img_p = dataset/item
            out_name = item.replace('/','_')
            cmd = 'cp '+img_p+'  '+dump_base/'img'/out_name
            os.system(cmd)
        if cp_gt:
            gt_p = dataset / item
            gt_p =gt_p.replace(shader,'depth')
            out_name = item.replace('/', '_')
            cmd = 'cp ' + gt_p + '  ' + dump_base / 'depth' / out_name
            os.system(cmd)


def chg_cmap():
    pass
    dir = Path('/home/roit/bluep2/test_out/mc/apr/mcv5-sildurs-e-10k-12345-s/depth')
    files = dir.files()
    for file in tqdm(files):
        if '.png' not in file:
            continue
        img = plt.imread(file)
        img = np.mean(img[:, :], axis=2)
        img= 1-img
        img = cv2.resize(img,(388,288))
        plt.imsave(file,img,cmap='magma')
        # plt.imsave(file,img)

    print('ok')

def kitti():
    dataset = Path("/media/roit/970evo/home/roit/datasets/kitti")

    wk_root = Path('/home/roit/aws/aprojects/xdr94_mono2')
    root = wk_root / 'splits/eigen/test_files.txt'

    out_path = wk_root / 'eigen_imgs'
    out_path.mkdir_p()
    files = readlines(root)
    for item in tqdm(files):
        dir,pre,num,lr = re.split(' |/',item)
        out_name = pre +'_'+ num+'_'+lr+'.png'
        cmd = 'cp '+ dataset/dir/pre/'image_02/data'/"{:010d}.png".format(int(num))+' '+out_path/out_name
        os.system(cmd)

    print('ok')

def extract_vsd_img():
    dataset = Path("/home/roit/datasets/VisDrone2")
    wk_root = Path('/home/roit/aws/aprojects/xdr94_mono2')
    root = wk_root / 'splits/visdrone/test_files.txt'
    img_dump = wk_root / 'visdrone_test_img'
    img_dump.mkdir_p()


    rel_paths = readlines(root)
    rel_paths.sort()
    for item in tqdm(rel_paths):
        seq,frame = item.split('/')
        img_p = dataset / seq / frame + '.jpg'
        out_name = item.replace('/', '_') + '.jpg'
        cmd = 'cp ' + img_p + '  ' + img_dump / out_name
        os.system(cmd)
def extract_vsd_img2():
        dataset = Path("/home/roit/datasets/VSD")
        wk_root = Path('/home/roit/aws/aprojects/xdr94_mono2')
        root = wk_root / 'splits/visdrone/test_files.txt'
        img_dump = wk_root / 'visdrone_test_img'
        img_dump.mkdir_p()

        rel_paths = readlines(root)
        rel_paths.sort()
        for item in tqdm(rel_paths):
            seq, frame = item.split('/')
            img_p = dataset / seq / frame + '.jpg'
            out_name = item.replace('/', '_') + '.jpg'
            cmd = 'cp ' + img_p + '  ' + img_dump / out_name
            os.system(cmd)



def resize():
    resize_h,resize_w = 192,640
    #imgs_p = Path("/home/roit/Desktop/img")
#    imgs_p = Path("./custom_test_img"
    imgs_p = Path("./mc_test_img")


    files=imgs_p.files()
    files.sort()
    for file in tqdm(files):
        img = cv2.imread(file)
        img_dump = cv2.resize(img,(resize_w,resize_h))
        cv2.imwrite(file,img_dump)



def extract_kitti_gt():
    path = Path("./splits/eigen/gt_depths.npz")
    gt = np.load(path, allow_pickle=True)
    print(path.exists())


if __name__ == '__main__':
    # extract_MC()
    chg_cmap()
    #extract_vsd_img()
    #extract_kitti_gt()
    #kitti()
    #resize()
