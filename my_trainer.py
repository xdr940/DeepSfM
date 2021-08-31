# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import time
import datetime
from path import Path
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import random
from utils.official import *
from utils.img_process import tensor2array
from kitti_utils import *
from networks.layers import *

from datasets import KITTIRAWDataset
# from datasets import KITTIOdomDataset
from datasets import MCDataset,VSDataset
from datasets import CustomMonoDataset
import networks
from utils.logger import TermLogger
from utils.assist import model_init,reframe
import torch


seed = 127
#random seed
np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# random.seed(seed)
#
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled= True

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

def L1_loss(pred,depth_gt):
    mask = depth_gt > depth_gt.min()
    mask *= depth_gt < depth_gt.max()
    abs_diff = torch.abs(depth_gt - pred)
    l1_loss = abs_diff[mask]  # [b,1,h,w]
    return l1_loss

def L2_loss(pred,depth_gt):
    mask = depth_gt > depth_gt.min()
    mask *= depth_gt < depth_gt.max()
    abs_sq = (depth_gt - pred)**2
    l2_loss = abs_sq[mask]  # [b,1,h,w]
    return l2_loss


class Trainer:
    def __init__(self, options,settings):

        def dataset_init(dataset_opt):

            # datasets setting
            datasets_dict = {"kitti": KITTIRAWDataset,
                             #"kitti_odom": KITTIOdomDataset,
                             "mc": MCDataset,
                             "custom_mono": CustomMonoDataset}

            if dataset_opt['type'] in datasets_dict.keys():
                dataset = datasets_dict[dataset_opt['type']]  # 选择建立哪个类，这里kitti，返回构造函数句柄
            else:
                dataset = CustomMonoDataset


            split_path = Path(dataset_opt['split']['path'])
            train_path = split_path / dataset_opt['split']['train_file']
            val_path = split_path / dataset_opt['split']['val_file']
            data_path = Path(dataset_opt['path'])

            feed_height = dataset_opt['to_height']
            feed_width = dataset_opt['to_width']


            batch_size = dataset_opt['batch_size']
            num_workers = dataset_opt['num_workers']


            train_filenames = readlines(train_path)
            val_filenames = readlines(val_path)
            img_ext = '.png'

            num_train_samples = len(train_filenames)
            # train loader
            train_dataset = dataset(  # KITTIRAWData
                data_path = data_path,
                filenames = train_filenames,
                height=feed_height,
                width=feed_width,
                frame_sides=self.frame_sides,#kitti[0,-1,1],mc[-1,0,1]
                num_scales = 4,
                mode="train"
                # img_ext='.png'
            )
            train_loader = DataLoader(  # train_datasets:KITTIRAWDataset
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            # val loader
            val_dataset = dataset(
                data_path=data_path,
                filenames=val_filenames,
                height=feed_height,
                width=feed_width,
                frame_sides = self.frame_sides,
                num_scales = 4,
                mode="val",
                img_ext=img_ext)

            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True)

            self.val_iter = iter(val_loader)
            print("Using split:{}, {}, {}".format(split_path,
                                                                        dataset_opt['split']['train_file'],
                                                                        dataset_opt['split']['val_file']
                                                                        ))
            print("There are {:d} training items and {:d} validation items".format(
                len(train_dataset), len(val_dataset)))

            return train_loader, val_loader

        #self.opt = options
        self.start_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.checkpoints_path = Path(options['log_dir'])/self.start_time
        self.frame_sides = options['frame_sides']# all around
        self.frame_prior,self.frame_now,self.frame_next = self.frame_sides
        self.scales = options['model']['scales']
        self.model_mode = options['model']['mode']
        self.framework = options['model']['framework']

        self.feed_width = options['dataset']['to_width']
        self.feed_height = options['dataset']['to_height']
        self.full_height = options['dataset']['full_height']
        self.full_width = options['dataset']['full_width']

        self.min_depth = options['min_depth']
        self.max_depth = options['max_depth']

        self.tb_log_frequency = options['tb_log_frequency']
        self.weights_save_frequency = options['weights_save_frequency']
        #save model and events


        #args assert
        self.device = torch.device(options['model']['device'])
        self.models, self.model_optimizer,self.model_lr_scheduler = model_init(options['model'])
        self.train_loader, self.val_loader = dataset_init(options['dataset'])
        self.dataset_type = options['dataset']['type']
        self.metrics = {"abs_rel": 0.0,
                        "sq_rel": 0.0,
                        "rmse": 0.0,
                        "rmse_log": 0.0,
                        "a1": 0.0,
                        "a2": 0.0,
                        "a3": 0.0
                        }
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
            h = options['dataset']['to_height'] // (2 ** scale)
            w = options['dataset']['to_width'] // (2 ** scale)

            self.layers['back_proj_depth'][scale] = BackprojectDepth(options['dataset']['batch_size'], h, w)
            self.layers['back_proj_depth'][scale].to(self.device)

            self.layers['project_3d'][scale] = Project3D(options['dataset']['batch_size'], h, w)
            self.layers['project_3d'][scale].to(self.device)

        #compute_depth_metrics




        #print("Training model named:\t  ", self.opt.model_name)
        print('--> frame sides:{}'.format(self.frame_sides))
        print("traing files are saved to: ", options['log_dir'])
        print("Training is using: ", self.device)
        print("start time: ",self.start_time)
        os.system('cp {} {}/train_settings.yaml'.format(settings, self.checkpoints_path))

        #self.save_opts()

        #custom

    def compute_losses(self,inputs, outputs):



        scales = self.scales
        frame_sides=self.frame_sides



        losses = {}
        total_loss = 0

        for scale in scales:
            loss = 0

            source_scale = 0

            disp = outputs[("disp", self.frame_now, scale)]
            color = inputs[("color", self.frame_now, scale)]
            target = inputs[("color", self.frame_now, source_scale)]

            # reprojection_losses
            reprojection_losses = []
            for frame_id in [self.frame_prior,self.frame_next]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(compute_reprojection_loss(self.layers['ssim'], pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # identity_reprojection_losses
            identity_reprojection_losses = []
            for frame_id in [self.frame_prior,self.frame_next]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    compute_reprojection_loss(self.layers['ssim'], pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            #
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties# 花书p149 向输入添加方差极小的噪声等价于 对权重施加范数惩罚
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            # erro_maps = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)  # b4hw

            # --------------------------------------------------------------
            map_34, idxs_1 = torch.min(reprojection_loss, dim=1)# w.o identical mask
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
    def compute_losses_spv(self,inputs, outputs):

        scales = self.scales



        losses = {}
        loss_scales = 0
        height,width = inputs[('color',0,0)].shape[2:]
        depth_gt_feed = F.interpolate(inputs["depth_gt"],[height,width], mode="bilinear", align_corners=False)

        for scale in scales:
            loss = 0
            #color = inputs[("color", 0, scale)]

            disp = outputs[("disp", 0, scale)]
            disp_full = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)
            _,depth  = disp_to_depth(disp_full)
            if scale ==0:
                outputs[("depth",0,scale)] = depth

            to_optimise =  L1_loss(disp_full, 1/(depth_gt_feed+1e-7))

            #to_optimise = (depth_gt_feed - depth).abs()**2


            loss += to_optimise.mean()


            mean_disp = disp_full.mean(2, True).mean(3, True)
            norm_disp = disp_full / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, 1/depth)

            loss += 0.1 * smooth_loss / (2 ** scale)

            loss_scales += loss
            losses["loss/{}".format(scale)] = loss

        loss_scales /= len(scales)
        losses["loss"] = loss_scales
        return  losses
    def compute_losses_rebuild(self,inputs,outputs):
        #TODO
        pass
    def generate_images_pred(self,inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary as color_identity.
        global inputs:
            self.scales
            self.height
            self.width
            self.min_depth
            self.max_depth
            self.frame_sides
            self.device

        """
        scales = self.scales
        frame_sides = self.frame_sides
        height = self.feed_height
        width = self.feed_width
        backproject_depth = self.layers['back_proj_depth']
        project_3d = self.layers['project_3d']


        for scale in scales:
            # get depth

            disp = outputs[("disp", 0, scale)]

            disp = F.interpolate(
                disp, [height, width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp,min_depth=self.min_depth,max_depth=self.max_depth)
            #depth = disp2depth(disp)

            outputs[("depth", 0, scale)] = depth

            for i, frame_side in enumerate([frame_sides[0],frame_sides[2]]):
                T = outputs[("cam_T_cam", 0, frame_side)]

                # from the authors of https://arxiv.org/abs/1712.00175

                cam_points = backproject_depth[source_scale](
                    depth,
                    inputs[("inv_K", source_scale)]
                )  # D@K_inv
                pix_coords = project_3d[source_scale](
                    cam_points,
                    inputs[("K", source_scale)],
                    T
                )  # K@D@K_inv

                outputs[("sample", frame_side, scale)] = pix_coords  # rigid_flow

                outputs[("color", frame_side, scale)] = F.grid_sample(inputs[("color", frame_side, source_scale)],
                                                                    outputs[("sample", frame_side, scale)],
                                                                    padding_mode="border", align_corners=True)

                outputs[("color_identity", frame_side, scale)] = inputs[("color", frame_side, source_scale)]

    #1. forward pass1, more like core
    def batch_process(self, model_mode,framework,inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        #self.device
        #self.models
        #self.generate_images_pred
        #self.compute_losses




        #device
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)


        outputs={}


        #depth pass
        colors = reframe(mode=model_mode[0],inputs=inputs,frame_sides=self.frame_sides)

        if framework == 'shared':
            features = self.models["encoder"](colors)#0:1611,1:1676

        elif framework == 'ind':
            features = self.models["depth_encoder"](colors)#0:1611,1:1676

        features = tuple(features)#0:2522, 1:5232
        disp = self.models["depth"](*features)


        outputs[("disp", 0, 0)] = disp[0]#0:3808
        outputs[("disp", 0, 1)] = disp[1]
        outputs[("disp", 0, 2)] = disp[2]
        outputs[("disp", 0, 3)] = disp[3]


        #pose pass
        poses=None
        if framework == 'shared':
            poses = self.models['pose'](*features)
        elif framework =='ind':
            colors = reframe(mode=model_mode[2], inputs=inputs, frame_sides=self.frame_sides)
            if model_mode[3] =='3dcnn':
                poses = self.models['pose'](colors)
            else:
                features = self.models['pose_encoder'](colors)
                poses = self.models['pose'](*features)


            #
        elif framework == "spv":
            losses = self.compute_losses_spv(inputs, outputs)

        elif framework == "rebuild":
            pass

        if framework in ['shared','ind']:
                #pose pass
            cam_T_cam = transformation_from_parameters(
                poses[:, 0,0,:3].unsqueeze(1), poses[:, 0,0,3:].unsqueeze(1), invert=True
            )  # b44
            outputs[("cam_T_cam", self.frame_now, self.frame_prior)] = cam_T_cam

            cam_T_cam = transformation_from_parameters(
                poses[:, 1,0,:3].unsqueeze(1), poses[:, 1,0,3:].unsqueeze(1), invert=False
            )  # b44
            outputs[("cam_T_cam", self.frame_now, self.frame_next)] = cam_T_cam

            self.generate_images_pred(inputs, outputs)  # 0:5629 # outputs get depth 0 0
            losses = self.compute_losses(inputs, outputs)  # 0:8561




        return outputs, losses


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
        mask = depth_gt > depth_gt.min()
        mask*=depth_gt< depth_gt.max()
        # garg/eigen crop#????
        if dataset_type =='kitti':#val_dataset, 由于gt缺失, 上半部分不参与
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        elif dataset_type =='mc':
            pass

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        '''
        for evaluation we multiply the predicted depth maps by a scalar 
        s_ that matches the median with the ground-truth, i.e. s_ = median(D_gt )/median(D_pred )
        #移动分布中心? move to the center of gt's distribution 
        '''
        depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)
        depth_gt = torch.clamp(depth_gt,min=min_depth,max = max_depth)

        metrics = compute_depth_errors(depth_gt, depth_pred)
        # for k, v in metrics.items():
        #     metrics[k] = np.array(v.cpu())
        return metrics



    def tb_log(self, mode,metrics, inputs=None, outputs=None, losses=None,add_img=True):

        """Write an event to the tensorboard events file
            scalers and images are saved same time, bad logic
        global inputs:
            self.step
            self.writers
            self.batch_size
            self.scales
            self.frame_sides
            self.mask
         """
        log_loss = ['loss']
        log_metrics=[
            "abs_rel",
            "sq_rel",
            "rmse",
            "rmse_log",
            "a1",
            "a2",
            "a3"]
        log_scales=[0]
        #log_frame_sides=[-1,0,1]
        log_frame_sides=[self.frame_prior,self.frame_now,self.frame_next]




        writer = self.writers[mode]
        if losses!=None:
            for k, v in losses.items():
                if k in log_loss:
                    writer.add_scalar("{}".format(k), float(v), self.step)
        if metrics!=None and   "depth_gt" in inputs.keys():
            for k,v in metrics.items():
                if k in log_metrics:
                    if k in ['a1','a2','a3']:
                        writer.add_scalar("acc/{}".format(k), v, self.step)
                    elif k in ['abs_rel','sq_rel','rmse','rmse_log']:
                        writer.add_scalar("err/{}".format(k), v, self.step)


        if add_img and ( inputs!=None or outputs!=None):
            b=0# int(random.random()*8)

            for s in log_scales:
                #color adde
                for frame_side in log_frame_sides:
                    if frame_side==0:
                        writer.add_image(
                            "color/{}".format(frame_side),
                            inputs[("color", frame_side, s)][b].data, self.step
                        )
                        if "depth_gt" in inputs.keys():
                            writer.add_image(
                                "gt/{}".format(frame_side),
                                normalize_image(inputs[("depth_gt")][b].data), self.step
                            )
                    else:
                        writer.add_image(
                            "color_pred/{}".format(frame_side, s, b),
                            outputs[("color", frame_side, s)][b].data, self.step
                        )

                #disp add
                writer.add_image(
                    "outputs/{}".format('disp'),
                    normalize_image(outputs[("disp", 0,s)][b]), self.step
                )
                writer.add_image(
                    "outputs/{}".format('depth'),
                    normalize_image(outputs[("depth", 0, s)][b]), self.step
                )



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
            try:
                outputs, losses = self.batch_process(self.model_mode,self.framework,inputs)#
            except:
                print('->batch process error')
                continue
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space

            #
            self.logger.train_logger_update(batch= batch_idx,time = duration,dict=losses)

            #val, and terminal_val_log, and tb_log
            if "depth_gt" in inputs:
                # train_set validate
                self.metrics.update(self.compute_depth_metrics(inputs, outputs, dataset_type=self.dataset_type))

            phase0 = batch_idx % self.tb_log_frequency == 0 and self.step < 1000
            phase1 = self.step % 100 == 0 and self.step > 1000 and self.step < 10000
            phase2 = self.step % 1000 == 0 and self.step > 10000 and self.step < 100000

            if phase0 or phase1 or phase2:


                self.tb_log(mode='train',
                            metrics=self.metrics,
                            inputs=inputs,
                            outputs=outputs,
                            losses=losses,
                            add_img=True
                            )
            else:
                self.tb_log(mode='train',
                            metrics=self.metrics,
                            inputs=inputs,
                            outputs=outputs,
                            losses=losses,
                            add_img=False
                            )


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
                                        dict=self.metrics
                                        )

        for epoch in range(opts['epoch']):
            epc_st = time.time()
            self.epoch_process()
            duration = time.time() - epc_st
            self.logger.epoch_logger_update(epoch=epoch+1,
                                            time=duration,
                                            dict=self.metrics
                                            )
            if (epoch + 1) % opts['weights_save_frequency'] == 0 and epoch >=opts['model_first_save']:


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

        val_batch_idx = self.val_iter._rcvd_idx

        time_st = time.time()
        outputs, losses = self.batch_process(self.model_mode,self.framework,inputs)



        duration =time.time() -  time_st
        self.logger.valid_logger_update(batch=val_batch_idx,
                                        time=duration,
                                        dict=losses
                                        )
        self.metrics.update(self.compute_depth_metrics(inputs, outputs, dataset_type=self.dataset_type))


        phase0 = val_batch_idx % self.tb_log_frequency == 0 and self.step < 1000
        phase1 = self.step % 100 == 0 and self.step > 1000 and self.step < 10000
        phase2 = self.step % 1000 == 0 and self.step > 10000 and self.step < 100000

        if phase0 or phase1 or phase2:

            self.tb_log(mode='val',
                        metrics=self.metrics,
                        inputs=inputs,
                        outputs=outputs,
                        losses=losses,
                        add_img=True
                        )
        else:
            self.tb_log(mode='val',
                        metrics=self.metrics,
                        inputs=inputs,
                        outputs=outputs,
                        losses=losses,
                        add_img=False
                        )




        del inputs, outputs, losses
        set_mode(self.models,'train')


