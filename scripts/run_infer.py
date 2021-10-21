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
from utils.official import compute_errors
from utils.assist import reframe
import matplotlib.pyplot as plt
from utils.official import np_normalize_image
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


def model_init(model_path,mode):
    encoder_path = model_path['encoder']
    decoder_path = model_path['depth']

    #model init
    encoder = getEncoder(components=mode)
    depth_decoder = getDepthDecoder(components='default',mode='test')

    #encoder dict updt
    encoder_dict = torch.load(encoder_path)
    encoder_dict = dict_update(encoder_dict)
    decoder_dict = torch.load(decoder_path)


    #load encoder dict
    model_dict = encoder.state_dict()
    model_dict_ = {k: v for k, v in encoder_dict.items() if k in model_dict}
    encoder.load_state_dict(model_dict_)

    #load decoder dict
    model_dict = depth_decoder.state_dict()
    decoder_dict_ = {k: v for k, v in decoder_dict.items() if k in model_dict}
    depth_decoder.load_state_dict(decoder_dict_)



    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()
    return encoder,depth_decoder




@torch.no_grad()
def evaluate(opts):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = opts['min_depth']
    MAX_DEPTH = opts['max_depth']

    data_path = opts['dataset']['path']
    batch_size = opts['dataset']['batch_size']

    num_workers = opts['dataset']['num_workers']
    feed_height = opts['feed_height']
    feed_width = opts['feed_width']
    full_width = opts['dataset']['full_width']
    full_height = opts['dataset']['full_height']

    out_dir = Path(opts['out_dir'])
    out_dir.mkdir_p()
    sub_dirs = opts['sub_dirs']
    for item in sub_dirs:
        (out_dir/item).mkdir_p()

    # metric_mode = opts['metric_mode']



    #这里的度量信息是强行将gt里的值都压缩到和scanner一样的量程， 这样会让值尽量接近度量值
    #但是对于


    data_path = Path(opts['dataset']['path'])
    lines = Path(opts['dataset']['split']['path'])/opts['dataset']['split']['test_file']
    model_path = opts['model']['load_paths']
    encoder_mode = opts['model']['encoder_mode']
    frame_sides = opts['frame_sides']
    # frame_prior,frame_now,frame_next =  opts['frame_sides']
    encoder,decoder = model_init(model_path,mode=encoder_mode)
    file_names = readlines(lines)

    print('-> dataset_path:{}'.format(data_path))
    print('-> model_path')
    for k,v in opts['model']['load_paths'].items():
        print('\t'+str(v))

    print("-> data split:{}".format(lines))
    print('-> total:{}'.format(len(file_names)))

    if opts['dataset']['type']=='mc':
        dataset = datasets.MCDataset(data_path=data_path,
                                       filenames=file_names,
                                       height=feed_height,
                                       width=feed_width,
                                       frame_sides=frame_sides,
                                     num_scales=1,
                                     mode="test")
    elif opts['dataset']['type']=='kitti':

        dataset = datasets.KITTIRAWDataset (  # KITTIRAWData
            data_path = data_path,
            filenames=file_names,
            height=feed_height,
            width=feed_width,
            frame_sides=frame_sides,
            num_scales=1,
            mode="test"
        )
    elif opts['dataset']['type']=='custom_mono':
        dataset = datasets.CustomMonoDataset(
            data_path=data_path,
            filenames=file_names,
            height=feed_height,
            width=feed_width,
            frame_sides=frame_sides,
            num_scales=1,
            mode='test'
        )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
    pred_depths=[]
    gt_depths = []
    disps = []
    idx=0
    for data in tqdm(dataloader):


        input_color = reframe(encoder_mode,data,frame_sides=frame_sides,key='color')
        input_color = input_color.cuda()


        features = encoder(input_color)
        disp = decoder(*features)



        # depth_gt = data['depth_gt']

        pred_disp, pred_depth = disp_to_depth(disp,min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
        #pred_depth = disp2depth(disp)





        if "depth" in sub_dirs:
            pred_depth = pred_depth.cpu()[:, 0].numpy()[0]
            depth = cv2.resize(pred_depth, (full_width, full_height))
            depth = np_normalize_image(depth)
            cv2.imwrite(out_dir/"depth"/file_names[idx].replace('/','_'),depth*255)

        if "disp" in sub_dirs:
            pred_disp = pred_disp.cpu()[:, 0].numpy()[0]
            disp = cv2.resize(pred_disp, (full_width, full_height))
            disp = np_normalize_image(disp)

            cv2.imwrite(out_dir/"disp"/file_names[idx].replace('/','_'),disp*255)



        idx+=1







if __name__ == "__main__":

    opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/fpv_infer.yaml').read_yaml()
    # opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/mc_eval.yaml').read_yaml()


    evaluate(opts)
