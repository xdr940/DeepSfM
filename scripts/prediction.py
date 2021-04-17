from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets.mc_dataset import relpath_split

from networks.encoders import getEncoder
from networks.decoders import getDepthDecoder

from networks.layers import disp2depth,disp_to_depth
from utils.official import readlines
import datasets
import networks
from tqdm import  tqdm
from path import Path
from utils.yaml_wrapper import YamlHandler
from utils.official import compute_errors
import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.

#
# def compute_errors(gt, pred):
#     """Computation of error metrics between predicted and ground truth depths
#     input HxW,HxW
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
    encoder = getEncoder(model_mode=mode)
    depth_decoder = getDepthDecoder(model_mode=1,mode='test')

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

def dataset_init():
    # dataloader

    pass
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
def prediction(opts):
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
    metric_mode = opts['metric_mode']

    framework_mode = opts['model']['mode']


    #这里的度量信息是强行将gt里的值都压缩到和scanner一样的量程， 这样会让值尽量接近度量值
    #但是对于


    data_path = Path(opts['dataset']['path'])
    lines = Path(opts['dataset']['split']['path'])/opts['dataset']['split']['test_file']
    model_path = opts['model']['load_paths']
    model_mode = opts['model']['mode']
    frame_sides = opts['frame_sides']
    out_dir_base = Path(opts['out_dir_base'])

    # frame_prior,frame_now,frame_next =  opts['frame_sides']
    encoder,decoder = model_init(model_path,mode=model_mode)
    file_names = readlines(lines)

    print('-> dataset_path:{}'.format(data_path))
    print('-> model_path')
    for k,v in opts['model']['load_paths'].items():
        print('\t'+str(v))

    print("-> metrics mode: {}".format(metric_mode))
    print("-> data split:{}".format(lines))
    print('-> total:{}'.format(len(file_names)))

    file_names.sort()
    #prediction loader
    # test_files = []
    # for base in file_names:
    #     test_files.append(data_path/base)
    # test_files.sort()


    if opts['dataset']['type']=='mc':
        dataset = datasets.MCDataset(data_path=data_path,
                                       filenames=file_names,
                                       height=feed_height,
                                       width=feed_width,
                                       frame_sides=frame_sides,
                                     num_scales=1,
                                     mode="prediction")
    elif opts['dataset']['type']=='kitti':

        dataset = datasets.KITTIRAWDataset (  # KITTIRAWData
            data_path = data_path,
            filenames=file_names,
            height=feed_height,
            width=feed_width,
            frame_sides=frame_sides,
            num_scales=1,
            mode="prediction"
        )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
    out_shows=[]

    if opts['out_dir']:
        out_dir = out_dir_base/opts['out_dir']
    else:
        out_dir = out_dir_base/data_path.stem
    out_dir.mkdir_p()
    for data in tqdm(dataloader):


        input_color = input_frames(data,mode=framework_mode,frame_sides=frame_sides)



        features = encoder(input_color)
        disp = decoder(*features)




        pred_disp, pred_depth = disp_to_depth(disp,min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)

        out_show = pred_disp
        out_show = out_show.cpu()[:,0].numpy()

        out_shows.append(out_show)


    for idx,item in enumerate(out_shows):


        depth_name = file_names[idx].replace('/', '_').replace('.png','depth')
        idx += 1
        plt.imsave(out_dir/depth_name+'{}'.format('.png'),item[0],cmap='magma')






    # preds_resized=[]
    # for item in pred_depths:
    #     pred_resized = cv2.resize(item, (full_width,full_height))
    #     preds_resized.append(np.expand_dims(pred_resized,axis=0))
    # preds_resized = np.concatenate(preds_resized,axis=0)











if __name__ == "__main__":

    # opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/kitti_eval.yaml').read_yaml()
    opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/mc_prediction.yaml').read_yaml()


    prediction(opts)
