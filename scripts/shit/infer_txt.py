# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
from path import Path
import numpy as np
import PIL.Image as pil
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.official import readlines
import torch
from torchvision import transforms
import networks
from networks.layers import PhotometricError,transformation_from_parameters,BackprojectDepth,Project3D,disp_to_depth
from utils.masks import VarMask,MeanMask,IdenticalMask,float8or

from datasets.kitti_dataset import KITTIRAWDataset
from datasets.visdrone_dataset import VSDataset
from datasets.mc_dataset import MCDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from opts.olds.run_from_txt_opts import run_from_txt_opts


#parse_args_run_from_txt  as parse_args
@torch.no_grad()
def main(args):
    """Function to predict for a single image or folder of images
    """
    print(args.dataset_path)
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")



    #download_model_if_doesnt_exist(args.model_path,args.model_name)

    model_path = Path(args.model_path)/ args.model_name
    if not model_path.exists():
        print(model_path+" does not exists")

    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

#1 LOADING PRETRAINED MODEL
    #1.1 encoder
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    #1.2 decoder
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

#2. FINDING INPUT IMAGES

    dataset_path = Path(args.dataset_path)

    #files
    root = Path(os.path.dirname(__file__))
    txt = root/'splits'/args.split/args.txt_files
    print('-> inference file: ',txt)
    rel_paths = readlines(txt)
    #out
    if args.out_path !=None:
        out_path  =Path(args.out_path)
    else:
        out_path = Path('./'+dataset_path.stem+'_out')
    out_path.mkdir_p()

    files=[]
    #rel_paths 2 paths
    if args.split in ['custom','custom_lite','eigen','eigen_zhou']:#kitti
        for item in  rel_paths:
            item = item.split(' ')
            if item[2]=='l':camera ='image_02'
            elif item[2]=='r': camera= 'image_01'
            files.append(dataset_path/item[0]/camera/'data'/"{:010d}.png".format(int(item[1])))
    elif args.split =='mc':
        for item in  rel_paths:
            #item = item.split('/')
            files.append(item)
    elif args.split =='visdrone'or 'visdrone_lite':
        for item in rel_paths:
            item = item.split('/')
            files.append(dataset_path / item[0] / item[1]+'.jpg')
    else :
        for item in rel_paths:
            item = item.split('/')
            files.append(dataset_path / item[0] / item[1]+'.jpg')

#2.1

    cnt=0
#3. PREDICTING ON EACH IMAGE IN TURN
    print('\n-> inference '+args.dataset_path)
    files.sort()
    for  image_path in tqdm(files):



        # Load image and preprocess

        if args.split =='mc':
            input_image = pil.open(dataset_path/image_path+'.png').convert('RGB')
        else:
            input_image = pil.open(image_path).convert('RGB')

        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)#torch.Size([1, 3, 192, 640])
        features = encoder(input_image)#a list from 0 to 4
        outputs = depth_decoder(features)# dict , 4 disptensor
        cnt+=1
        disp = outputs[("disp", 0)]# has a same size with input
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        #if args.out_name=='num':
        if args.split=='eigen' or args.split=='custom':
            output_name = str(image_path).split('/')[-4]+'_{}'.format(image_path.stem)
        elif args.split =='mc':
            block,p,color,frame =image_path.split('/')
            output_name = str(image_path).replace('/','_')+'.png'
        elif args.split=='visdrone' or args.split=='visdrone_lite':
            output_name = image_path.relpath(dataset_path).strip('.jpg').replace('/','_')
            pass
        elif args.split=='custom_mono':
            output_name = image_path.relpath(dataset_path).strip('.jpg').replace('/','_')
        else:
            output_name = image_path.relpath(dataset_path).strip('.jpg').replace('/','_')




        if args.npy_out:
            name_dest_npy = os.path.join(out_path, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        name_dest_im =Path(out_path)/"{}.png".format(output_name)
        plt.imsave(name_dest_im, disp_resized_np, cmap='magma', vmax=vmax)

    print(cnt)

    print('\n-> Done,save at '+args.out_path)

def main_with_masks(args):
    """Function to predict for a single image or folder of images
    """
    print(args.dataset_path)
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    out_path = Path(args.out_path)
    out_path.mkdir_p()
    dirs ={}
    for mask in args.results:
        dirs[mask] = (out_path/mask)
        (out_path/mask).mkdir_p()

    print('-> split:{}'.format(args.split))
    print('-> save to {}'.format(args.out_path))


    if args.split in['custom','custom_lite','eigen','eigen_zhou']:
        feed_height = 192
        feed_width = 640
        min_depth=0.1
        max_depth=80
        full_height=375
        full_width=1242
        dataset = KITTIRAWDataset

    elif args.split in["visdrone","visdrone_lite"]:
        feed_width=352
        feed_height=192
        min_depth=0.1
        max_depth=255
        dataset = VSDataset
    elif args.split in['mc','mc_lite']:
        feed_height=288
        feed_width=384
        min_depth=0.1
        max_depth=255
        dataset=MCDataset

    feed_height=192
    feed_width=640

    backproject_depth = BackprojectDepth(1,feed_height,feed_width).to(device)

    project_3d = Project3D(1,feed_height,feed_width)

    photometric_error = PhotometricError()

    txt_files = args.txt_files
    #data
    test_path = Path(args.wk_root) / "splits" / args.split / txt_files
    test_filenames = readlines(test_path)
    if args.as_name_sort:#按照序列顺序名字排列
        test_filenames.sort()
    #check filenames:
    i=0
    for i,item in enumerate(test_filenames):
        #item = test_filenames[i]
        if args.split in ['eigen','custom','custom_lite','eigen_zhou']:
            dirname,frame,lr = test_filenames[i].split()
            files =  (Path(args.dataset_path)/dirname/'image_02/data').files()
            files.sort()
            min =int(files[0].stem)
            max = int(files[-1].stem)
            if int(frame)+args.frame_ids[0]<=min or int(frame)+args.frame_ids[-1]>=max:
                test_filenames[i]=''
        if args.split in ['mc','mc_lite']:#虽然在split的时候已经处理过了
            block,trajactory,color,frame = test_filenames[i].split('/')
            files = (Path(args.dataset_path) / block/trajactory/color).files()
            files.sort()
            min = int(files[0].stem)
            max = int(files[-1].stem)
            if int(frame) + args.frame_ids[0] <= min or int(frame) + args.frame_ids[-1] >= max:
                test_filenames[i] = ''
            pass
        if args.split in ['visdrone','visdrone_lite']:#虽然在split的时候已经处理过了
            dirname,frame = test_filenames[i].split('/')
            files =  (Path(args.dataset_path)/dirname).files()
            files.sort()
            min = int(files[0].stem)
            max = int(files[-1].stem)
            if int(frame) + args.frame_ids[0] <= min or int(frame) + args.frame_ids[-1] >= max:
                test_filenames[i] = ''


    while '' in test_filenames:
        test_filenames.remove('')



    test_dataset = dataset(  # KITTIRAWData
        args.dataset_path,
        test_filenames,
        feed_height,
        feed_width,
        args.frame_ids,
        1,
        is_train=False,
        img_ext=args.ext)

    test_loader = DataLoader(  # train_datasets:KITTIRAWDataset
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False)


    print('->items num: {}'.format(len(test_loader)))

    #layers


    #download_model_if_doesnt_exist(args.model_path,args.model_name)

    model_path = Path(args.model_path)/ args.model_name
    if not model_path.exists():
        print(model_path+" does not exists")

    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")



#1 LOADING PRETRAINED MODEL
    #1.1 encoder
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    #1.2 decoder
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()


    #paths
    pose_encoder_path = Path(model_path) / "pose_encoder.pth"
    pose_decoder_path = Path(model_path) / 'pose.pth'

    # 2.1 pose encoder
    print("   Loading pretrained pose encoder")

    pose_encoder = networks.ResnetEncoder(18, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))


    pose_encoder.to(device)
    pose_encoder.eval()


    # 2.2 pose decoder
    print("   Loading pretrained decoder")
    pose_decoder = networks.PoseDecoder(
        num_ch_enc=pose_encoder.num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=2)

    pose_loaded_dict = torch.load(pose_decoder_path, map_location=device)
    pose_decoder.load_state_dict(pose_loaded_dict)

    pose_decoder.to(device)
    pose_decoder.eval()
    source_scale = 0
    scale=0
    for batch_idx, inputs in tqdm(enumerate(test_loader)):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(device)
        features = encoder(inputs[("color",0,0)])  # a list from 0 to 4

        outputs = depth_decoder(features)  # dict , 4 disptensor

        disp = outputs[("disp", 0)]  # has a same size with input

        #disp_resized = torch.nn.functional.interpolate(disp, (full_height, full_width), mode="bilinear", align_corners=False)

        _, depth = disp_to_depth(disp, min_depth, max_depth)





        for f_i in [args.frame_ids[0],args.frame_ids[-1]]:

            if f_i < 0:
                pose_inputs = [inputs[("color",f_i,0)],inputs[("color",0,0)]]
            else:
                pose_inputs = [inputs[("color",0,0)], inputs[("color",f_i,0)]]
            pose_inputs = torch.cat(pose_inputs, 1)
            features = pose_encoder(pose_inputs)
            axisangle, translation = pose_decoder([features])

            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0],  invert=(f_i < 0))  # b44
            T = outputs[("cam_T_cam", 0, f_i)]

            cam_points = backproject_depth(depth, inputs[("inv_K", 0)])  # D@K_inv
            pix_coords = project_3d(cam_points, inputs[("K", 0)], T)  # K@D@K_inv





            outputs[("sample", f_i, 0)] = pix_coords  # rigid_flow

            outputs[("color", f_i, 0)] = F.grid_sample(inputs[("color", f_i, 0)],
                                                                outputs[("sample", f_i, 0)],
                                                                padding_mode="border")
            # output"color" 就是i-warped

            # add a depth warp
            outputs[("color_identity", f_i, 0)] = inputs[("color", f_i, 0)]


        target = inputs[("color", 0, 0)]

        reprojection_losses = []
        for frame_id in [args.frame_ids[0],args.frame_ids[-1]]:
            pred = outputs[("color", frame_id, 0)]
            reprojection_losses.append(photometric_error.run(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)



        identity_reprojection_losses = []
        for frame_id in [args.frame_ids[0],args.frame_ids[-1]]:
            pred = inputs[("color", frame_id, source_scale)]
            identity_reprojection_losses.append(photometric_error.run(pred, target))
        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        erro_maps = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)  # b4hw



        identical_mask = IdenticalMask(erro_maps)
        identical_mask = identical_mask[0].detach().cpu().numpy()


        save_name = test_filenames[batch_idx].replace('/','_')
        save_name = save_name.replace('l','')
        save_name = save_name.replace('r','')
        save_name = save_name.replace(' ','')




        if "identical_mask" in args.results:
            plt.imsave(dirs['identical_mask'] / "{}.png".format(save_name), identical_mask)



        if "depth" in args.results:
            # Saving colormapped depth image
            disp_np = disp[0,0].detach().cpu().numpy()
            vmax = np.percentile(disp_np, 95)
            plt.imsave(dirs['depth']/"{}.png".format(save_name), disp_np,cmap='magma',vmax=vmax)



        if "mean_mask" in args.results:
            mean_mask = MeanMask(erro_maps)
            mean_mask = mean_mask[0].detach().cpu().numpy()
            plt.imsave(dirs['mean_mask']/"{}.png".format(save_name), mean_mask,cmap='bone')



        if "identical_mask" in args.results:
            identical_mask = IdenticalMask(erro_maps)
            identical_mask = identical_mask[0].detach().cpu().numpy()
            plt.imsave(dirs['identical_mask']/"{}.png".format(save_name), identical_mask,cmap='bone')


        if "var_mask" in args.results:
            var_mask = VarMask(erro_maps)
            var_mask = var_mask[0].detach().cpu().numpy()
            plt.imsave(dirs["var_mask"]/"{}.png".format(save_name),var_mask,cmap='bone')

        if "final_mask" in args.results:
            identical_mask = IdenticalMask(erro_maps)
            mean_mask = MeanMask(erro_maps)
            var_mask = VarMask(erro_maps)
            final_mask = float8or(mean_mask* identical_mask, var_mask)
            final_mask = final_mask[0].detach().cpu().numpy()
            plt.imsave(dirs["final_mask"] / "{}.png".format(save_name), final_mask,cmap='bone')



if __name__ == '__main__':
    options = run_from_txt_opts()
    #main_with_masks(options.parse())
    main(options.parse())

