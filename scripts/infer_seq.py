# /bin/python
# 所有连续真

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets.mc_dataset import relpath_split

from networks.encoders import getEncoder
from networks.depth_decoder import getDepthDecoder

from networks.layers import disp2depth,disp_to_depth
from utils.official import readlines
import datasets
import networks
from tqdm import  tqdm
from path import Path
from utils.yaml_wrapper import YamlHandler
from utils.assist import dataset_init_infer_dir,model_init_infer,reframe,model_init
from utils.official import np_normalize_image
import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)






def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def dict_update(dict):

    keys = list(dict.keys()).copy()
    for key in keys:
        if 'encoder.' in key:
            new_key = key.replace('encoder.','')
            dict[new_key] =  dict.pop(key)

    return dict


def input_frames(data,mode,frame_sides):
    if mode=="3din":
        input = torch.cat([data["color", frame_sides[0], 0].unsqueeze(dim=2),
                             data["color", frame_sides[1], 0].unsqueeze(dim=2),
                             data["color", frame_sides[2], 0].unsqueeze(dim=2)],
                            dim=2)

    elif mode =='3in':
        input = torch.cat([data["color", frame_sides[0], 0],
                             data["color", frame_sides[1], 0],
                             data["color", frame_sides[2], 0]],
                            dim=1)

    elif mode=='1in':
        input = data["color", 0, 0]




    return input.cuda()

def post_press(out_put):
    pass


@torch.no_grad()
def prediction(configs):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = configs['min_depth']
    MAX_DEPTH = configs['max_depth']

    data_path = configs['dataset']['path']
    batch_size = configs['dataset']['batch_size']

    num_workers = configs['dataset']['num_workers']
    feed_height = configs['feed_height']
    feed_width = configs['feed_width']
    full_width = configs['dataset']['full_width']
    full_height = configs['dataset']['full_height']
    frame_sides = configs['frame_sides']
    paradigm = configs['model']['paradigm']
    components = configs['model']['components']
    load_paths = configs['model']['load_paths']
    dump_path = Path(configs['dump_path'])
    dump_path.mkdir_p()
    #这里的度量信息是强行将gt里的值都压缩到和scanner一样的量程， 这样会让值尽量接近度量值
    #但是对于

    data_infer_loader = dataset_init_infer_dir(configs)

    models = model_init_infer(configs)


    for idx, inputs in tqdm(enumerate(data_infer_loader)):


        #depth pass
        colors = reframe(component=components[0],inputs=inputs,frame_sides=frame_sides)
        if colors ==None:
            continue
        if paradigm == 'shared':
            features = models["encoder"](colors)#0:1611,1:1676

        elif paradigm == 'ind':
            features = models["depth_encoder"](colors)#0:1611,1:1676

        features = tuple(features)#0:2522, 1:5232
        disp = models["depth"](*features)




        # depth_gt = data['depth_gt']

        pred_disp, pred_depth = disp_to_depth(disp[0], min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
        # pred_depth = disp2depth(disp)
        pred_depth = pred_depth.cpu()[:, 0].numpy()[0]
        depth = cv2.resize(pred_depth, (full_width, full_height))
        depth = np_normalize_image(depth)
        # plt.imsave(dump_path /"depth_{:05d}.png".format(idx), depth * 255,cmap='plasma')

        pred_disp = pred_disp.cpu()[:, 0].numpy()[0]
        disp = cv2.resize(pred_disp, (full_width, full_height))
        disp = np_normalize_image(disp)
        plt.imsave(dump_path /"disp_{:05d}.png".format(idx), disp * 255,cmap='plasma')








if __name__ == "__main__":

    # configs = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/configs/kitti_eval.yaml').read_yaml()
    configs = YamlHandler('/home/roit/aws/aprojects/DeepSfM/configs/kitti_infer_seq.yaml').read_yaml()


    prediction(configs)
