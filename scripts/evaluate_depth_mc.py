from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets.mc_dataset import relpath_split

from networks.new_encoders import getEncoder
from networks.depth_decoder import getDepthDecoder

from networks.layers import disp2depth
from utils.official import readlines
import datasets
import networks
from tqdm import  tqdm
from path import Path
from utils.yaml_wrapper import YamlHandler

import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
SCALE_FACTOR = 800.


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    input HxW,HxW
    """
    #pred+=1e-4
    #gt+=1e-4
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)


    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



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


def model_init(model_path):
    encoder_path = model_path['encoder']
    decoder_path = model_path['depth']


    encoder = getEncoder(model_mode=0)
    depth_decoder = getDepthDecoder(model_mode=1,mode='test')

    encoder_dict = torch.load(encoder_path)
    encoder_dict = dict_update(encoder_dict)
    decoder_dict = torch.load(decoder_path)



    model_dict = encoder.state_dict()
    model_dict_ = {k: v for k, v in encoder_dict.items() if k in model_dict}
    encoder.load_state_dict(model_dict_)

    model_dict = depth_decoder.state_dict()
    decoder_dict_ = {k: v for k, v in decoder_dict.items() if k in model_dict}
    depth_decoder.load_state_dict(decoder_dict_)



    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()
    return encoder,depth_decoder

def dataset_init():
    # dataloader

    pass


@torch.no_grad()
def evaluate(opts):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-4
    MAX_DEPTH = 1.

    data_path = opts['dataset']['path']
    num_workers = opts['dataset']['num_workers']
    feed_height = opts['feed_height']
    feed_width = opts['feed_width']
    full_width = opts['dataset']['full_width']
    full_height = opts['dataset']['full_height']


    #这里的度量信息是强行将gt里的值都压缩到和scanner一样的量程， 这样会让值尽量接近度量值
    #但是对于


    data_path = Path(opts['dataset']['path'])
    lines = Path(opts['dataset']['split']['path'])/'test2.txt'
    model_path = opts['model']['load_paths']
    encoder,decoder = model_init(model_path)
    file_names = readlines(lines)

    dataset = datasets.MCDataset(data_path,
                                       file_names,
                                       feed_height,
                                       feed_width,
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)
    pred_depths=[]
    gt_depths = []
    disps = []
    for data in tqdm(dataloader):
        input_color = data[("color", 0, 0)].cuda()
        features = encoder(input_color)
        disp = decoder(*features)



        depth_gt = data['depth_gt']

        #pred_disp, pred_depth = disp_to_depth(disp,min_depth=0.1, max_depth=576.0)
        pred_depth = disp2depth(disp)

        pred_depth = pred_depth.cpu().numpy()
        depth_gt = depth_gt.cpu().numpy()
        disp = disp.cpu().numpy()

        pred_depths.append(pred_depth)
        gt_depths.append(depth_gt)
        disps.append(disp)
    gt_depths = np.concatenate(gt_depths, axis=0)
    gt_depths = gt_depths.squeeze(axis=1)


    pred_depths = np.concatenate(pred_depths,axis=0)
    pred_depths = pred_depths.squeeze(axis=1)





    disps = np.concatenate(disps,axis=0)
    disps = disps.squeeze(axis=1)

    preds_resized=[]
    for item in pred_depths:
        pred_resized = cv2.resize(item, (full_width,full_height))
        preds_resized.append(np.expand_dims(pred_resized,axis=0))
    preds_resized = np.concatenate(preds_resized,axis=0)


    radomed = gt_depths

    metrics = []
    ratios=[]
    cnt=0
    for gt,pred in zip(gt_depths,preds_resized):

        gt[gt < MIN_DEPTH] = MIN_DEPTH
        gt[gt > MAX_DEPTH] = MAX_DEPTH

        ratio = np.median(gt) / np.median(pred)  # 中位数， 在eval的时候， 将pred值线性变化，尽量能使与gt接近即可
        ratios.append(ratio)
        pred *= ratio

        pred[pred < MIN_DEPTH] = MIN_DEPTH
        pred[pred > MAX_DEPTH] = MAX_DEPTH



        metric = compute_errors(pred, gt)
        metrics.append(metric)

    metrics = np.array(metrics)
    print( np.median(metrics, axis=0))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print("\n Scaling ratios | med: {:0.3f} | std: {:0.3f}\n".format(med, np.std(ratios / med)))

    pass










if __name__ == "__main__":

    opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/mc_eval.yaml').read_yaml()

    evaluate(opts)
