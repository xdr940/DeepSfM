from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that run_infer_opts.py resides in

class train_opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 training options")
        self.parser.add_argument("--num_epochs", type=int, help="number of epochs", default=10)
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=8)  #
        self.parser.add_argument("--weights_save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        #splits
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="/home/roit/datasets/splits/eigen_zhou_std"
                                 )
        self.parser.add_argument('--train_files',
                                 default='train_files.txt',
                                 )
        self.parser.add_argument('--val_files',
                                 default='val_files.txt')



        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 default='/home/roit/models/monodepth2_official/mono_640x192',
                                 help="name of model to load, if not set, train from imgnt pretrained")


        #dataset
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default='kitti',
                                 # "kitti",
                                 # "kitti_odom",
                                 # "kitti_depth",
                                 # "mc",
                                 # 'visdrone',
                                 # 'custom_mono'
                                 )
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/home/roit/datasets/kitti/'
                                )
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='/home/roit/models/monodepth2/kitti'
                                  )
        self.parser.add_argument('--pose_arch',default='en_decoder',choices=['en_decoder','share-encoder','posecnn'])

        self.parser.add_argument("--masks",
                                 default=['identity_selection',
                                          'ind_mov',
                                          'poles',
                                          'final_selection',
                                          'var_mask',
                                          'mean_mask',
                                          'ind_mov',
                                          'map_12',
                                          'map_34'
                                         ])
        self.parser.add_argument('--root',type=str,
                                 help="project root",
                                 default='/home/roit/aws/aprojects/xdr94_mono2')



        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 default=True,
                                 action="store_true")


        self.parser.add_argument("--height",type=int,help="model input image height",
                                 #default=288#mc
                                 #default=192#kitti
                                 default=192#visdrone
                                 )
        self.parser.add_argument("--width",type=int,help="model input image width",
                                 #default=384#MC
                                 default=640#kitti
                                 #default=352
                                 )
        #只是为了计算metrics用的， 如果vsd之类的没有， 可以不填
        self.parser.add_argument("--full_height",type=int,
                                 default=375#kitti
                                 #default = 600#mc
                                 #default=1071#vs
                                 #default=1080#custom mono
                                 )

        self.parser.add_argument("--full_width",type=int,
                                 default=1242#kitti
                                 #default = 800#mc
                                 #default=1904#vs
                                 #default=1920#custom_mono
                                 )

        self.parser.add_argument("--disparity_smoothness",type=float,help="disparity smoothness weight",default=0.1)

        self.parser.add_argument("--scales",nargs="+",type=int,help="scales used in the loss",default=[0, 1, 2, 3])

        self.parser.add_argument("--min_depth",type=float,help="minimum depth",default=0.1)#这里度量就代表m
        self.parser.add_argument("--max_depth",type=float,help="maximum depth",
                                 default=80.0#kitti
                                 #default=80.0#visdrone
                                 #default = 255.0
                                 #default = 800.0
                                )


        #self.parser.add_argument("--use_stereo",help="if set, uses stereo pair for training",action="store_true")
        self.parser.add_argument("--frame_ids",nargs="+",type=int,help="frames to load",
                                 default=[0,-1,1]#visdrone
                                #default = [0, -1, 1]

        )

        # OPTIMIZATION options

        self.parser.add_argument("--learning_rate",type=float,help="learning rate",default=1e-4)
        self.parser.add_argument("--start_epoch",type=int,help="for subsequent training",
                                 #default=10,
                                 default=0,

                                 )

        self.parser.add_argument("--scheduler_step_size",type=int,help="step size of the scheduler",default=15)

        # LOADING args for subsquent training or train from pretrained/scratch

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load, for training or test",
                                 default=["encoder",
                                          "depth",
                                          "pose_encoder",
                                          "pose"
                                          #"posecnn"
                                          ])

        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch or subsequent training from last",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--encoder_path",
                                 type=str,
                                 help="pretrained from here",
                                 default="/home/roit/models/torchvision/official/resnet18-5c106cde.pth",
                                 )
        self.parser.add_argument("--posecnn_path",
                                 type=str,
                                 help="pretrained from here",
                                 #default="/home/roit/models/SCBian_official/k_pose.tar",
                                 default=None,

                                 )




        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)



        # LOGGING options
        self.parser.add_argument("--tb_log_frequency",
                                 type=int,
                                 help="number of batches(step) between each tensorboard log",
                                 default=12)

    def args(self):
        self.options = self.parser.parse_args()
        return self.options
