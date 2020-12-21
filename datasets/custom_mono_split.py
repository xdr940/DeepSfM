
from path import Path
import path
from random import random
import random
import argparse
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--dataset_path', type=str,default='/home/roit/datasets/Binjiang/',help='path to a test image or folder of images')
    parser.add_argument("--txt_style",default='custom_mono',choices=['mc','visdrone','custom_mono'])
    parser.add_argument('--out_path', type=str,default=None,help='path to a test image or folder of images')
    parser.add_argument("--num",default=1000,type=str)
    parser.add_argument("--proportion",default=[0.9,0.05,0.05],help="train, val, test")
    parser.add_argument("--out_name",default=None)
    parser.add_argument("--out_dir",default='./custom_mono')

    return parser.parse_args()


def writelines(list,path):
    lenth = len(list)
    with open(path,'w') as f:
        for i in range(lenth):
            if i == lenth-1:
                f.writelines(str(list[i]))
            else:
                f.writelines(str(list[i])+'\n')

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
def generate_mc(args):
    '''

    :param args:
    :return:none , output is  a dir includes 3 .txt files
    '''
    [train_,val_,test_] = args.proportion

    if train_+val_+test_-1.>0.01:#delta
        print('erro')
        return


    dataset_path = Path(args.dataset_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir_p()
    train_txt_p = out_dir/'train_files.txt'
    val_txt_p = out_dir/'val_files.txt'
    test_txt_p = out_dir/'test_files.txt'

    sequeces = dataset_path.dirs()
    sequeces.sort()#seqs
    item_list=[]#

    for seq in sequeces:
        imgs = seq.files()
        imgs.sort()
        for p in imgs:
            item_list.append(p)

    random.shuffle(item_list)
    total_list=[]
    for idx,img_p in enumerate(item_list):
        frame_num  =int(img_p.stem)

        if frame_num > 1 and frame_num+1 <len(img_p.parent.files()):
            item = img_p.relpath('/home/roit/datasets/Binjiang').strip('.jpg')
            total_list.append(item)



    print('1/2')



    train_bound = int(args.num *args.proportion[0])
    val_bound = int(args.num *args.proportion[1])+train_bound
    test_bound = int(args.num *args.proportion[2])+val_bound

    writelines(total_list[:train_bound],train_txt_p)
    writelines(total_list[train_bound:val_bound],val_txt_p)
    writelines(total_list[val_bound:test_bound],test_txt_p)

    print('ok')











if  __name__ == '__main__':
    options = parse_args()
    generate_mc(options)