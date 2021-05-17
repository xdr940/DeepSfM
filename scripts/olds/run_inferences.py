# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import cv2
import os
from path import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from networks.layers import disp_to_depth
from opts.olds.run_infer_opts import run_inference_opts



def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    print(args.image_path)
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(args.model_path)
    #download_model_if_doesnt_exist(args.model_path,args.model_name)

    model_path = os.path.join(args.model_path, args.model_name)

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

    in_path = Path(args.image_path)
    if args.out_path !=None:
        out_path  =Path(args.out_path)
    else:
        out_path = Path('./'+in_path.stem+'_out')

    out_path.mkdir_p()


#3. PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for  image_path in tqdm(in_path.files()):



            # Load image and preprocess

            input_image = cv2.imread(image_path)#.convert('RGB')
            input_image = input_image[70:600,:,:]
            original_height, original_width, _ = input_image.shape
            input_image = cv2.resize(input_image,(640,192))

            # input_image = pil.open(image_path).convert('RGB')
            #original_width, original_height = input_image.size
            #input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)#torch.Size([1, 3, 192, 640])
            features = encoder(input_image)#a list from 0 to 4
            outputs = depth_decoder(features)# dict , 4 disptensor

#            disp = outputs[("disp", 0,0)]# has a same size with input
            disp = outputs[("disp", 0)]# has a same size with input

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = image_path.stem
            if args.npy_out:
                name_dest_npy = os.path.join(out_path, "{}_disp.npy".format(output_name))
                scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            name_dest_im = os.path.join(out_path, "{}_disp.png".format(output_name))
            plt.imsave(name_dest_im, disp_resized_np, cmap='magma', vmax=vmax)



    print('-> Done!')


if __name__ == '__main__':
    options = run_inference_opts()
    opts = options.parse()
    test_simple(opts)
