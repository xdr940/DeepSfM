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


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.start_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.checkpoints_path = Path(self.opt.log_dir)/self.start_time
        #save model and events


        #args assert
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"


    #models
        #details
        self.models = {}#dict
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2


        #depth encoder
        if self.opt.load_weights_folder:
            pass
        else:
            print("depth encoder pretrained or scratch: "+self.opt.weights_init)
            print("depth encoder load:"+self.opt.encoder_path)
        self.models["encoder"] = networks.ResnetEncoder(
            num_layers=self.opt.num_layers,
            pretrained=self.opt.weights_init == "pretrained",
            encoder_path=self.opt.encoder_path)
        self.models["encoder"].to(self.device)

        #depth decoder
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)



        if self.opt.pose_arch=='en_decoder':
            # pose encoder
            if self.opt.load_weights_folder:
                pass
            else:
                print("pose encoder pretrained or scratch: " + self.opt.weights_init)
                print("pose encoder load:" + self.opt.encoder_path)
            self.models["pose_encoder"] = networks.ResnetEncoder(
                                            self.opt.num_layers,
                                            self.opt.weights_init == "pretrained",
                                            num_input_images=self.num_pose_frames,
                                            encoder_path=self.opt.encoder_path)
            self.models["pose_encoder"].to(self.device)

            # pose decoder
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["pose"].to(self.device)


        #posecnn
        elif self.opt.pose_arch=='posecnn':
            if self.opt.posecnn_path:

                print("posecnn pretrained or scratch: " + self.opt.weights_init)
                print("posecnn load:" + self.opt.posecnn_path)
            self.models['posecnn'] = networks.PoseNet().to(self.device)





    #trainale params
        parameters_to_train=[]
        for k,v in self.models.items():
            parameters_to_train+= list(v.parameters())
        self.model_optimizer = optim.Adam(parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
#load model
        if self.opt.load_weights_folder is not None:
            print("load model from {} instead pretrain".format(self.opt.load_weights_folder))
            self.load_model()




        # datasets setting
        datasets_dict = {"kitti": KITTIRAWDataset,
                         "kitti_odom": KITTIOdomDataset,
                         "mc":MCDataset,
                         "custom_mono":CustomMonoDataset,
                         "visdrone":VSDataset}
        if self.opt.dataset in datasets_dict.keys():

            self.dataset = datasets_dict[self.opt.dataset]#选择建立哪个类，这里kitti，返回构造函数句柄
        else:
            self.dataset = CustomMonoDataset


        train_path = Path(self.opt.split)/options.train_files
        val_path = Path(self.opt.split)/options.val_files


        train_filenames = readlines(train_path)
        val_filenames = readlines(val_path)
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        #train loader
        train_dataset = self.dataset(#KITTIRAWData
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=True,
            img_ext=img_ext)
        self.train_loader = DataLoader(#train_datasets:KITTIRAWDataset
            dataset=train_dataset,
            batch_size= self.opt.batch_size,
            shuffle= False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        #val loader
        val_dataset = self.dataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,
            img_ext=img_ext)

        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(self.checkpoints_path/mode)

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:{}, train_files:{}, val_files:{}".format( self.opt.split,self.opt.train_files,self.opt.val_files))
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        #print("Training model named:\t  ", self.opt.model_name)
        print("traing files are saved to: ", self.opt.log_dir)
        print("Training is using: ", self.device)
        print("start time: ",self.start_time)


        self.save_opts()

        #custom

        self.logger = TermLogger(n_epochs=self.opt.num_epochs,
                            train_size=len(self.train_loader),
                            valid_size=len(self.val_loader))
        self.logger.reset_epoch_bar()

        self.metrics = {}





    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    #1. forward pass1, more like core
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

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
        outputs.update(self.predict_poses(inputs, features))        #outputs get 3 new values


        #4.
        self.generate_images_pred(inputs, outputs)#outputs get
        losses = self.compute_losses_f(inputs, outputs)

        return outputs, losses

    #2. called by 1
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        write outputs
        """
        outputs = {}

        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}



        for f_i in self.opt.frame_ids[1:]:

            # To maintain ordering we always pass frames in temporal order

            #map concat
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

        #pose en-decoder
            #encoder map
            if self.opt.pose_arch=='en_decoder':
                features_mid = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]#pose_inputs list of 2 [b,3,h,w] i.e b,6,h,w

                #decoder
                axisangle, translation = self.models["pose"](features_mid)#b213,b213



                    #outputs[("axisangle", 0, f_i)] = axisangle#没用？
                    #outputs[("translation", 0, f_i)] = translation#没用？

                # Invert the matrix if the frame id is negative
                cam_T_cam= transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                )#b44
                outputs[("cam_T_cam", 0, f_i)] = cam_T_cam
            elif self.opt.pose_arch=='posecnn':


                pose = self.models['posecnn'](pose_inputs[0],pose_inputs[1])

                cam_T_cam = transformation_from_parameters(
                    pose[:,:3].unsqueeze(1), pose[:,3:].unsqueeze(1), invert=(f_i < 0)
                )  # b44
                outputs[("cam_T_cam", 0, f_i)] = cam_T_cam


        return outputs

    #3.
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary as color_identity.
        """
        for scale in self.opt.scales:
            # get depth

            disp = outputs[("disp", 0,scale)]

            disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):


                T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175


                cam_points = self.backproject_depth[source_scale](depth,
                                                                inputs[("inv_K", source_scale)])# D@K_inv
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)# K@D@K_inv


                outputs[("sample", frame_id, scale)] = pix_coords#rigid_flow

                outputs[("color", frame_id, scale)]= F.grid_sample(inputs[("color", frame_id, source_scale)],
                                                                    outputs[("sample", frame_id, scale)],
                                                                    padding_mode="border",align_corners=True)


                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

    #4.
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)#[b,1,h,w]

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss#[b,1,h,w]



    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []


            source_scale = 0

            disp = outputs[("disp", 0,scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)


            identity_reprojection_loss = identity_reprojection_losses




            reprojection_loss = reprojection_losses

            # add random numbers to break ties# 花书p149 向输入添加方差极小的噪声等价于 对权重施加范数惩罚
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)


            to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_losses_f(self, inputs, outputs):
        """通过var 计算移动物体
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            #reprojection_losses
            reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            #identity_reprojection_losses
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)


            #
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties# 花书p149 向输入添加方差极小的噪声等价于 对权重施加范数惩罚
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            erro_maps = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)#b4hw

# --------------------------------------------------------------
            map_34, idxs_1 = torch.min(reprojection_loss, dim=1)
            #map_34 = torch.mean(reprojection_loss, dim=1)

            #var_mask = VarMask(erro_maps)
            #mean_mask = MeanMask(erro_maps)
            #identity_selection = IdenticalMask(erro_maps)

            #final_mask = float8or(float8or(1 - mean_mask, identity_selection), var_mask)#kitti
            #final_mask = float8or((mean_mask * identity_selection),var_mask)#vsd
            #to_optimise = map_34 * identity_selection
            #to_optimise = map_34 * final_mask
            to_optimise = map_34

            #outputs["identity_selection/{}".format(scale)] = identity_selection.float()
            #outputs["mean_mask/{}".format(scale)] = mean_mask.float()

            #outputs["var_mask/{}".format(scale)] = var_mask.float()

            #outputs["final_selection/{}".format(scale)] = final_mask.float()

# ----------------------------------------

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses


    def compute_depth_metrics(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance

        in1 outputs[("depth", 0, 0)]
        in2 inputs["depth_gt"]
        out1 losses
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [self.opt.full_height, self.opt.full_width], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop#????
        crop_mask = torch.zeros_like(mask)
        if self.opt.dataset =='kitti':
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        metrics={}
        for i, metric in enumerate(self.depth_metric_names):
            metrics[metric] = np.array(depth_errors[i].cpu())
        return  metrics



    def terminal_log(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def tb_log(self, mode, inputs=None, outputs=None, losses=None,metrics=None):
        """Write an event to the tensorboard events file
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
                    for frame_id in self.opt.frame_ids:
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

                    '''        
                    if "identity_selection/{}".format(s) in outputs.keys():
                        img = tensor2array(outputs["identity_selection/{}".format(s)][j],colormap='bone')
                        writer.add_image(
                                "automask_{}/{}".format(s, j),
                                #outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                                img, self.step)#add 1,h,w

                    if "mo_ob/{}".format(s) in outputs.keys():
                        img = tensor2array(outputs["mo_ob/{}".format(s)][j],colormap='bone')
                        writer.add_image(
                                "mo_ob_{}/{}".format(s, j),
                                #outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                                img, self.step)#add 1,h,w
                    if "final_selection/{}".format(s) in outputs.keys():
                        img = tensor2array(outputs["final_selection/{}".format(s)][j],colormap='bone')
                        writer.add_image(
                                "final_selection_{}/{}".format(s, j),
                                #outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                                img, self.step)#add 1,h,w
                    '''

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = self.checkpoints_path/"models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(self.checkpoints_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """

        save_folder = self.checkpoints_path/"models"/"weights_{}".format(self.epoch)
        save_folder.makedirs_p()

        for model_name, model in self.models.items():
            save_path = save_folder/ "{}.pth".format(model_name)
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = save_folder/ "{}.pth".format("adam")
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from load_weights_folder
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
        print("Loading {} weights...".format(self.opt.models_to_load))

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    #main cycle
    def epoch_train(self):
        """Run a single epoch of training and validation
        """

        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            #model forwardpass
            outputs, losses = self.process_batch(inputs)#

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.tb_log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            #
            self.logger.train_logger_update(batch= batch_idx,time = duration,names=losses.keys(),values=[item.cpu().data for item in losses.values()])

            #val, and terminal_val_log, and tb_log
            if early_phase or late_phase:
                #self.terminal_log(batch_idx, duration, losses["loss"].cpu().data)


                #metrics={}
                #if "depth_gt" in inputs:
                #    metrics = self.compute_depth_metrics(inputs, outputs)
                self.tb_log(mode="train", inputs=inputs, outputs=outputs, losses =losses)#terminal log
                if "depth_gt" in inputs:
                    self.metrics = self.compute_depth_metrics(inputs, outputs)
                    self.tb_log(mode='train', metrics=self.metrics)


                self.val()


            self.step += 1

        self.model_lr_scheduler.step()

        self.logger.reset_train_bar()
        self.logger.reset_valid_bar()

            #record the metric

    #only 2 methods for public call
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        self.logger.epoch_logger_update(epoch=0,
                                        time=0,
                                        names=self.metrics.keys(),
                                        values=["{:.4f}".format(float(item)) for item in self.metrics.values()])

        for self.epoch in range(self.opt.start_epoch,self.opt.num_epochs):
            epc_st = time.time()
            self.epoch_train()
            duration = time.time() - epc_st
            self.logger.epoch_logger_update(epoch=self.epoch+1,
                                            time=duration,
                                            names=self.metrics.keys(),
                                            values=["{:.4f}".format(float(item)) for item in self.metrics.values()])
            if (self.epoch + 1) % self.opt.weights_save_frequency == 0 :
                self.save_model()

    @torch.no_grad()
    def val(self):
        """Validate the model on a single minibatch
        这和之前的常用框架不同， 之前是在train all batches 后再 val all batches，
        这里train batch 再 val batch（frequency）
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
        time_st = time.time()
        outputs, losses = self.process_batch(inputs)
        duration =time.time() -  time_st
        self.logger.valid_logger_update(batch=self.val_iter._rcvd_idx,
                                        time=duration*self.opt.tb_log_frequency,
                                        names=losses.keys(),
                                        values=[item.cpu().data for item in losses.values()])



        self.tb_log(mode="val", inputs=inputs, outputs=outputs, losses=losses)

        if "depth_gt" in inputs:
            metrics = self.compute_depth_metrics(inputs, outputs)
            self.tb_log(mode="val",metrics = metrics)
            del metrics

        del inputs, outputs, losses
        self.set_train()
