'''
only for kitti. coupled with export_gt_depth.py
'''

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader



from networks.encoders import getEncoder
from networks.decoders import getDepthDecoder



from networks.layers import disp_to_depth
from utils.official import readlines
import datasets
import networks
from tqdm import  tqdm
from path import Path
from utils.yaml_wrapper import YamlHandler
from utils.official import compute_errors

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
def dict_update(dict):

    keys = list(dict.keys()).copy()
    for key in keys:
        if 'encoder.' in key:
            new_key = key.replace('encoder.','')
            dict[new_key] =  dict.pop(key)

    return dict



# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


# def compute_errors(gt, pred):
#     """Computation of error metrics between predicted and ground truth depths
#
#     """
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25     ).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()
#
#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())
#
#     rmse_log = (np.log(gt) - np.log(pred)) ** 2
#     rmse_log = np.sqrt(rmse_log.mean())
#
#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#
#     sq_rel = np.mean(((gt - pred) ** 2) / gt)
#
#     return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

@torch.no_grad()
def evaluate(opts):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = opts['min_depth']
    MAX_DEPTH = opts['max_depth']

    data_path = opts['dataset']['path']
    num_workers = opts['dataset']['num_workers']
    feed_height = opts['feed_height']
    feed_width = opts['feed_width']
    full_width = opts['dataset']['full_width']
    full_height = opts['dataset']['full_height']
    batch_size = opts['dataset']['batch_size']
    #这里的度量信息是强行将gt里的值都压缩到和scanner一样的量程， 这样会让值尽量接近度量值
    #但是对于

    # if not opt.eval_mono or opt.eval_stereo:print("Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo")

    test_dir = Path(opts['dataset']['split']['path'])
    test_file = opts['dataset']['split']['test_file']

    # 1. load gt
    print('\n-> load gt:{}\n'.format(test_dir))
    gt_path = test_dir / "gt_depths.npz"
    gt_depths = np.load(gt_path,allow_pickle=True)
    gt_depths = gt_depths["data"]


    #2. load img data and predict, output is pred_disps(shape is [nums,1,w,h])

    # depth_eval_path = Path(opt.depth_eval_path)
    #
    # if not depth_eval_path.exists():print("Cannot find a folder at {}".format(depth_eval_path))
    #
    #
    # print("-> Loading weights from {}".format(depth_eval_path))


    #model loading
    filenames = readlines(test_dir/ test_file)
    encoder_path = opts['model']['load_paths']['encoder']
    decoder_path = opts['model']['load_paths']['depth']

    encoder = getEncoder(model_mode='1in')
    depth_decoder = getDepthDecoder(model_mode=1, mode='test')

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

    # dataloader
    dataset = datasets.KITTIRAWDataset(data_path,
                                       filenames,
                                       feed_height,
                                       feed_width,
                                       [0],
                                       4,
                                       is_train=False)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)


    pred_disps = []
    pred_depths=[]
    # gt_depths=[]

    print("\n-> Computing predictions with size {}x{}\n".format(
        encoder_dict['width'], encoder_dict['height']))

    #prediction
    for data in tqdm(dataloader):
        input_color = data[("color", 0, 0)].cuda()

        # if opt.post_process:
        #     # Post-processed results require each image to have two forward passes
        #     input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
        #eval 0
        output = depth_decoder(encoder(input_color))

        #eval 1
        pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
        pred_depth = pred_depth.cpu()[:, 0].numpy()
        #pred_depth = pred_depth.cpu()[:,0].numpy()
        # if opt.post_process:
        #     N = pred_disp.shape[0] // 2
        #     pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

        # depth_gt = data['depth_gt']
        # depth_gt = depth_gt.cpu().numpy()
        # gt_depths.append(depth_gt)
        pred_depths.append(pred_depth)
    #endfor
    pred_depths = np.concatenate(pred_depths)
    # gt_depths = np.concatenate(gt_depths)
    # gt_depths = gt_depths.squeeze(axis=1)




    metrics = []
    ratios = []

    for gt, pred in zip(gt_depths, pred_depths):
        gt_height, gt_width = gt.shape[:2]
        pred =  cv2.resize(pred, (gt_width, gt_height))
        #crop
        # if test_dir.stem == "eigen" or test_dir.stem == 'custom':#???,可能是以前很老的
        if opts['dataset']['type'] == "kitti":  # ???,可能是以前很老的
            mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt > 0

        pred = pred[mask]#并reshape成1d
        gt = gt[mask]

        ratio = np.median(gt) / np.median(pred)#中位数， 在eval的时候， 将pred值线性变化，尽量能使与gt接近即可
        ratios.append(ratio)
        pred *= ratio

        pred[pred < MIN_DEPTH] = MIN_DEPTH#所有历史数据中最小的depth, 更新,
        pred[pred > MAX_DEPTH] = MAX_DEPTH#...
        metric = compute_errors(gt, pred)
        metrics.append(metric)

    metrics = np.array(metrics)
    mean_metrics = np.mean(metrics, axis=0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_metrics.tolist()) + "\\\\")

    ratios = np.array(ratios)
    med = np.median(ratios)
    print("\n Scaling ratios | med: {:0.3f} | std: {:0.3f}\n".format(med, np.std(ratios / med)))
    #
    # mean_metrics = metrics.mean(0)
    # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    # print(("&{: 8.3f}  " * 7).format(*mean_metrics.tolist()) + "\\\\")
    # print("\n-> Done!")




if __name__ == "__main__":
    opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/kitti_eval.yaml').read_yaml()

    evaluate(opts)
