

import numpy as np
from torchvision import transforms
import random
import torch
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
import PIL.Image as pil


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_sides,
                 num_scales,
                 mode="train",
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames#list , like '2011_09_26/2011_09_26_drive_0001_sync 1 l'
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_sides = frame_sides

        self.mode = mode#unsuper train or evaluation
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness,
                self.contrast,
                self.saturation,
                self.hue
            )
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resizor = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resizor[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        #self.load_depth = False



    #private
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                type, side, _ = k
                for scale in range(self.num_scales):
                    inputs[(type, side, scale)] = self.resizor[scale](inputs[(type, side, scale - 1)])
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                type, side, scale= k
                inputs[(type, side, scale)] = self.to_tensor(f)
                if scale<=0 and color_aug:
                    inputs[(type + "_aug", side, scale)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item(inputs ) from the datasets as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk#1242x362
            0       images resized to (self.width,      self.height     )#640x192
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.mode=="train" and random.random() > 0.5
        do_flip = self.mode=="train" and random.random() > 0.5
        do_rotation = self.mode=="train" and random.random()>0.5

        split_line = self.filenames[index]


        #img sides
        for side in self.frame_sides:
            try:
                inputs[("color", side, -1)] = self.get_color(split_line, side,  do_flip)  # inputs得到scale == -1的前 中后三帧
            except:
                import os
                # os.system('clear')
                # print(split_line)
                exit(-1)

        # adjusting intrinsics to match each scale in the pyramid

        if self.mode=="train" or self.mode=="val":
            for scale in range(self.num_scales):
                K = self.K.copy()

                K[0, :] *= self.width // (2 ** scale)
                K[1, :] *= self.height // (2 ** scale)

                inv_K = np.linalg.pinv(K)

                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

            color_aug=None
            if do_color_aug:
                color_aug = transforms.ColorJitter.get_params(#对图像进行稍微处理，aug 但是要保证深度一致
                    self.brightness, self.contrast, self.saturation, self.hue)


            else:# self.mode=="val":
                color_aug = (lambda x: x)
        else: #test
            color_aug=None

        self.preprocess(inputs, color_aug)#scalse,aug generate to 38




        for i in self.frame_sides:
            if ("color", i, -1) in inputs.keys():
                del inputs[("color", i, -1)]#删除原分辨率图
            if ("color_aug", i, -1) in inputs.keys():
                del inputs[("color_aug", i, -1)]#删除原分辨率曾广图



        if self.load_depth:
            depth_gt = self.get_depth(split_line, 0,  do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))





        return inputs
        # 多态, 继承类用

    def get_color(self, line, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, line, side, do_flip):
        raise NotImplementedError