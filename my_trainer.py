# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import time
import datetime
from path import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json

from utils.official import *
from utils.img_process import tensor2array
from kitti_utils import *
from networks.layers import *

from datasets import KITTIRAWDataset
from datasets import KITTIOdomDataset
from datasets import MCDataset,VSDataset
from datasets import CustomMonoDataset
import networks
from utils.logger import TermLogger

def set_mode(models, mode):
    """Convert all models to training mode
    """
    for m in models.values():
        if mode == 'train':
            m.train()
        else:
            m.eval()
def compute_reprojection_loss(ssim, pred, target):
            """Computes reprojection loss between a batch of predicted and target images
            """
            abs_diff = torch.abs(target - pred)
            l1_loss = abs_diff.mean(1, True)  # [b,1,h,w]

            ssim_loss = ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

            return reprojection_loss  # [b,1,h,w]
class Trainer:
    def __init__(self, options):
        def model_init(mopt):
            # models
            # details
            models = {}  # dict
            self.scales = mopt['scales']
            self.num_input_frames = len(mopt['frame_idxs'])
            self.num_pose_frames = 2

            # depth encoder
            init_weight_path = Path(mopt['init_weight_path'])
            if mopt['init_weight_path']:
                print("depth encoder pretrained: " + self.opt.weights_init)
                print("depth encoder load:" + self.opt.encoder_path)
            else:
                print("depth encoder pretrained: " + self.opt.weights_init)


            #encoder
            models["encoder"] = networks.ResnetEncoder(
                num_layers=18,#resnet18
                pretrained=True,
                encoder_path=init_weight_path/'encoder.pth')
            models["encoder"].to(self.device)

            # depth decoder
            models["depth"] = networks.DepthDecoder(
                num_ch_enc = models["encoder"].num_ch_enc,
                scales=mopt['scales'])
            models["depth"].to(self.device)


            # pose encoder
            print("pose encoder pretrained or scratch: " + self.opt.weights_init)
            print("pose encoder load:" + self.opt.encoder_path)
            models["pose_encoder"] = networks.ResnetEncoder(
                num_layers=18,
                pretrained=True,
                num_input_images=2,
                encoder_path=init_weight_path/'pose_encoder')
            models["pose_encoder"].to(self.device)

            # pose decoder
            models["pose"] = networks.PoseDecoder(
                models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            models["pose"].to(self.device)


            # posecnn
            # print("posecnn pretrained or scratch: " + self.opt.weights_init)
            # print("posecnn load:" + self.opt.posecnn_path)
            # self.models['posecnn'] = networks.PoseNet().to(self.device)
            parameters_to_train = []
            for k, v in models.items():
                parameters_to_train += list(v.parameters())

            model_optimizer = optim.Adam(parameters_to_train, mopt['lr'])

            model_lr_scheduler = optim.lr_scheduler.StepLR(
                model_optimizer,
                mopt['scheduler_step_size'],
                0.1
            )#end models arch

            #model load
            if mopt['init_weight_path']:
                load_weights_folder = Path(mopt['init_weight_path'])

                for name in mopt['load_mnames']:
                    path = load_weights_folder / name + '.pth'
                    model_dict = self.models[name].state_dict()
                    pretrained_dict = torch.load(path)
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.models[name].load_state_dict(model_dict)
                print("Loading {} weights...".format(mopt['load_mnames']))

                # loading adam state
                optimizer_load_path = load_weights_folder / "adam.pth"
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)


            return models,model_optimizer,model_lr_scheduler
        def dataset_init(dopt):

            # datasets setting
            datasets_dict = {"kitti": KITTIRAWDataset,
                             "kitti_odom": KITTIOdomDataset,
                             "mc": MCDataset,
                             "custom_mono": CustomMonoDataset,
                             "visdrone": VSDataset}
            if dopt['type'] in datasets_dict.keys():
                dataset = datasets_dict[dopt['type']]  # 选择建立哪个类，这里kitti，返回构造函数句柄
            else:
                dataset = CustomMonoDataset


            split_path = Path(dopt['split']['path'])
            train_path = split_path / dopt['split']['train_file']
            val_path = split_path / dopt['split']['val_file']



            train_filenames = readlines(train_path)
            val_filenames = readlines(val_path)
            img_ext = '.png' if self.opt.png else '.jpg'

            num_train_samples = len(train_filenames)
            self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

            # train loader
            train_dataset = dataset(  # KITTIRAWData
                data_path = dopt['path'],
                filenames = train_filenames,
                height=self.opt.height,
                width=self.opt.width,
                frame_idxs=self.frame_idxs,
                num_scales = 4,
                is_train=True,
                img_ext='.png'
            )
            train_loader = DataLoader(  # train_datasets:KITTIRAWDataset
                dataset=train_dataset,
                batch_size=dopt['batch_size'],
                shuffle=False,
                num_workers=dopt['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            # val loader
            val_dataset = dataset(
                data_path=dopt['path'],
                filenames=val_filenames,
                height=self.opt.height,
                width=self.opt.width,
                frame_idxs = self.frame_idxs,
                num_scales = 4,
                is_train=False,
                img_ext=img_ext)

            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=dopt['batch_size'],
                shuffle=True,
                num_workers=dopt['num_workers'],
                pin_memory=True,
                drop_last=True)

            self.val_iter = iter(val_loader)
            print("Using split:{}, train_files:{}, val_files:{}".format(split_path,
                                                                        dopt['split']['train_file'],
                                                                        dopt['split']['val_file']
                                                                        ))
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))

            return train_loader, val_loader

        #self.opt = options
        self.start_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.checkpoints_path = Path(options['settings']['log_dir'])/self.start_time
        self.frame_idxs = options['frame_idxs']# all around
        self.scales = options['scales']
        self.feed_width = options['feed_width']
        self.feed_height = options['feed_height']

        self.full_height = options['full_height']
        self.full_width = options['full_width']

        self.min_depth = options['min_depth']
        self.max_depth = options['max_depth']

        self.tb_log_frequency = options['tb_log_frequency']
        self.weights_save_frequency = options['weights_save_frequency']

        #save model and events


        #args assert

        self.device = torch.device("cpu" if options['cuda'] else "cuda")
        self.models, self.model_optimizer,self.model_lr_scheduler = model_init(options['model'])
        self.train_loader, self.val_loader = dataset_init(options['dataset'])


        #
        self.logger = TermLogger(n_epochs=options['epoch'],
                                 train_size=len(self.train_loader),
                                 valid_size=len(self.val_loader))
        self.logger.reset_epoch_bar()







        #static define


        # tb_log
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(self.checkpoints_path/mode)


        #self.layers
        self.layers={}
        self.layers['ssim'] = SSIM()
        self.layers['ssim'].to(self.device)

        self.layers['back_proj_depth']={}
        self.layers['project_3d']={}

        #batch_process, generate_pred
        self.backproject_depth = {}
        self.project_3d = {}

        for scale in options['scales']:
            h = options['height'] // (2 ** scale)
            w = options['width'] // (2 ** scale)

            self.layers['back_proj_depth'][scale] = BackprojectDepth(options['batch_size'], h, w)
            self.layers['back_proj_depth'][scale].to(self.device)

            self.layers['project_3d'][scale] = Project3D(options['batch_size'], h, w)
            self.layers['project_3d'][scale].to(self.device)

        #compute_depth_metrics
        self.metrics = {"de/abs_rel":0.0,
                        "de/sq_rel":0.0,
                        "de/rms":0.0,
                        "de/log_rms":0.0,
                        "da/a1":0.0,
                        "da/a2":0.0,
                        "da/a3":0.0
                        }



        #print("Training model named:\t  ", self.opt.model_name)
        print("traing files are saved to: ", self.opt.log_dir)
        print("Training is using: ", self.device)
        print("start time: ",self.start_time)


        #self.save_opts()

        #custom

    def compute_losses_f(self,inputs, outputs):



        scales = self.scales
        frame_idxs=self.frame_idxs



        losses = {}
        total_loss = 0

        for scale in scales:
            loss = 0

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # reprojection_losses
            reprojection_losses = []
            for frame_id in self.frame_idxs[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(compute_reprojection_loss(self.ssim, pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # identity_reprojection_losses
            identity_reprojection_losses = []
            for frame_id in frame_idxs[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    compute_reprojection_loss(self.ssim, pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            #
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties# 花书p149 向输入添加方差极小的噪声等价于 对权重施加范数惩罚
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            erro_maps = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)  # b4hw

            # --------------------------------------------------------------
            map_34, idxs_1 = torch.min(reprojection_loss, dim=1)
            # map_34 = torch.mean(reprojection_loss, dim=1)

            # var_mask = VarMask(erro_maps)
            # mean_mask = MeanMask(erro_maps)
            # identity_selection = IdenticalMask(erro_maps)

            # final_mask = float8or(float8or(1 - mean_mask, identity_selection), var_mask)#kitti
            # final_mask = float8or((mean_mask * identity_selection),var_mask)#vsd
            # to_optimise = map_34 * identity_selection
            # to_optimise = map_34 * final_mask
            to_optimise = map_34

            # outputs["identity_selection/{}".format(scale)] = identity_selection.float()
            # outputs["mean_mask/{}".format(scale)] = mean_mask.float()

            # outputs["var_mask/{}".format(scale)] = var_mask.float()

            # outputs["final_selection/{}".format(scale)] = final_mask.float()

            # ----------------------------------------

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += 0.1 * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= len(scales)
        losses["loss"] = total_loss
        return losses

    def generate_images_pred(self,inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary as color_identity.
        global inputs:
            self.scales
            self.height
            self.width
            self.min_depth
            self.max_depth
            self.frame_idxs
            self.device

        """
        scales = self.scales
        frame_idxs = self.frame_idxs
        height = self.feed_height
        width = self.feed_width



        for scale in scales:
            # get depth

            disp = outputs[("disp", 0, scale)]

            disp = F.interpolate(
                disp, [height, width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(frame_idxs[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175

                cam_points = self.backproject_depth[source_scale](depth,
                                                                  inputs[("inv_K", source_scale)])  # D@K_inv
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)  # K@D@K_inv

                outputs[("sample", frame_id, scale)] = pix_coords  # rigid_flow

                outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)],
                                                                    outputs[("sample", frame_id, scale)],
                                                                    padding_mode="border", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

    #1. forward pass1, more like core
    def batch_process(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        def predict_poses(inputs, features):
            """Predict poses between input frames for monocular sequences.
            write outputs
            self inputs:
                self.frame_idxs
                self.models
                self.pose_arch


            """
            outputs = {}

            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input

            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_idxs}

            for f_i in range(len(self.frame_idxs)):

                # To maintain ordering we always pass frames in temporal order

                # map concat
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                # pose en-decoder
                # encoder map
                #pose_arch == 'en_decoder':


                features_mid = [
                    self.models["pose_encoder"](
                        torch.cat(pose_inputs, 1)
                    )
                ]  # pose_inputs list of 2 [b,3,h,w] i.e b,6,h,w

                # decoder
                axisangle, translation = self.models["pose"](features_mid)  # b213,b213

                # outputs[("axisangle", 0, f_i)] = axisangle#没用？
                # outputs[("translation", 0, f_i)] = translation#没用？

                # Invert the matrix if the frame id is negative
                cam_T_cam = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                )  # b44
                outputs[("cam_T_cam", 0, f_i)] = cam_T_cam

                #pose_arch == 'posecnn':
                # pose = self.models['posecnn'](pose_inputs[0], pose_inputs[1])
                #
                # cam_T_cam = transformation_from_parameters(
                #     pose[:, :3].unsqueeze(1), pose[:, 3:].unsqueeze(1), invert=(f_i < 0)
                # )  # b44
                # outputs[("cam_T_cam", 0, f_i)] = cam_T_cam

            return outputs

        #
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)


        #1. depth output
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # 关于output是由depth model来构型的
            # outputs only have disp 0,1,2,3

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        outputs[("disp", 0, 0)] = outputs[("disp", 0)]
        outputs[("disp", 0, 1)] = outputs[("disp", 1)]
        outputs[("disp", 0, 2)] = outputs[("disp", 2)]
        outputs[("disp", 0, 3)] = outputs[("disp", 3)]

        #2. mask

        #3. pose
        outputs.update(predict_poses(inputs, features))        #outputs get 3 new values


        #4.
        self.generate_images_pred(inputs, outputs)#outputs get
        losses = self.compute_losses_f(inputs, outputs)

        return outputs, losses

    #2. called by 1





    def compute_depth_metrics(self, inputs, outputs,dataset_type):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance

        in1 outputs[("depth", 0, 0)]
        in2 inputs["depth_gt"]
        out1 losses

        used by epoch
        """
        min_depth = self.min_depth
        max_depth = self.max_depth
        full_height = self.full_height
        full_width = self.full_width

        depth_pred_full = F.interpolate(outputs[("depth", 0, 0)], [full_height, full_width], mode="bilinear", align_corners=False)
        depth_pred = torch.clamp(
              depth_pred_full,
              min=min_depth,
              max=max_depth
          )
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop#????
        crop_mask = torch.zeros_like(mask)
        if dataset_type =='kitti':#val_dataset
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)


        metrics = compute_depth_errors(depth_gt, depth_pred)
        for k, v in metrics.items():
            metrics[k] = np.array(v.cpu())
        return metrics



    def tb_log(self, mode, inputs=None, outputs=None, losses=None,metrics=None):
        """Write an event to the tensorboard events file
        global inputs:
            self.step
            self.writers
            self.batch_size
            self.scales
            self.frame_idxs
            self.mask
         """
        writer = self.writers[mode]
        if losses!=None:
            for l, v in losses.items():
                writer.add_scalar("{}".format(l), v, self.step)
        if metrics!=None:
            for l,v in metrics.items():
                writer.add_scalar("{}".format(l), v, self.step)
        if inputs!=None and outputs!=None:
            for j in range(min(4, self.opt.batch_size)):  # write a maxminmum of four images
                for s in self.opt.scales:
                    #multi scale
                    if s !=0:
                        continue
                    #color add
                    for frame_id in self.frame_idxs:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                    #disp add
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", 0,s)][j]), self.step)
                    #mask add
                    for mask in self.opt.masks:
                        if mask+'/{}'.format(s) in outputs.keys():
                            img = tensor2array(outputs[mask+'/{}'.format(s)][j], colormap='bone')
                            writer.add_image(
                                mask+"{}/{}".format(s, j),
                                img, self.step)  # add 1,h,w


    #main cycle
    def epoch_process(self):
        """Run a single epoch of training and validation
        global_inputs:
            self.models
            self.motel_optimizer
            self.model_lr_scheduler
            self.step
            self.tb_log_frequency
            self.logger
        global_methods:
            self.batch_process
            self.tb_log
            self.compute_depth_metrics
            self.val
        """
        set_mode(self.models,'train')

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            #model forwardpass
            outputs, losses = self.batch_process(inputs)#

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.tb_log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            #
            self.logger.train_logger_update(batch= batch_idx,time = duration,names=losses.keys(),values=[item.cpu().data for item in losses.values()])

            #val, and terminal_val_log, and tb_log
            if early_phase or late_phase:


                if "depth_gt" in inputs:
                    #train_set validate
                    self.metrics = self.compute_depth_metrics(inputs, outputs,dataset_type='kitti')
                    self.tb_log(mode='train', metrics=self.metrics)


                self.val()


            self.step += 1

        self.model_lr_scheduler.step()
        self.logger.reset_train_bar()
        self.logger.reset_valid_bar()

            #record the metric

    #only 2 methods for public call
    def __call__(self,opts):

        """Run the entire training pipeline
        """




        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        self.logger.epoch_logger_update(epoch=0,
                                        time=0,
                                        names=self.metrics.keys(),
                                        values=["{:.4f}".format(float(item)) for item in self.metrics.values()])

        for epoch in range(opts['model']['epoch']):
            epc_st = time.time()
            self.epoch_process()
            duration = time.time() - epc_st
            self.logger.epoch_logger_update(epoch=epoch+1,
                                            time=duration,
                                            names=self.metrics.keys(),
                                            values=["{:.4f}".format(float(item)) for item in self.metrics.values()])
            if (epoch + 1) % opts['settings']['weights_save_frequency'] == 0 :


                save_folder = self.checkpoints_path / "models" / "weights_{}".format(epoch)
                save_folder.makedirs_p()

                for model_name, model in self.models.items():
                    save_path = save_folder / "{}.pth".format(model_name)
                    to_save = model.state_dict()
                    # if model_name == 'encoder':
                    #     # save the sizes - these are needed at prediction time
                    #     to_save['height'] = input_size['height']
                    #     to_save['width'] = input_size['width']
                    torch.save(to_save, save_path)
                #optimizer
                save_path = self.checkpoints_path/'models'/ "{}.pth".format("adam")
                torch.save(self.model_optimizer.state_dict(), save_path)



    @torch.no_grad()
    def val(self):
        """Validate the model on a single minibatch
        这和之前的常用框架不同， 之前是在train all batches 后再 val all batches，
        这里train batch 再 val batch（frequency）
        global inputs:
            self.models
            self.logger
        global methods:
            self.batch_process
            self.tb_log

        """
        set_mode(self.models,'eval')

        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()


        time_st = time.time()
        outputs, losses = self.batch_process(inputs)
        duration =time.time() -  time_st
        self.logger.valid_logger_update(batch=self.val_iter._rcvd_idx,
                                        time=duration*self.opt.tb_log_frequency,
                                        names=losses.keys(),
                                        values=[item.cpu().data for item in losses.values()])



        self.tb_log(mode="val", inputs=inputs, outputs=outputs, losses=losses)

        if "depth_gt" in inputs:
            #val_set validate
            self.metrics = self.compute_depth_metrics(inputs, outputs,dataset_type='kitti')
            self.tb_log(mode="val",metrics = self.metrics)

        del self.metrics
        del inputs, outputs, losses
        set_mode(self.models,'train')


