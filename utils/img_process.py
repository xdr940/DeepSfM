from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import numpy as np
import torch
np.seterr(divide='ignore',invalid='ignore')

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}



def tensor2array(tensor, max_value=None, colormap='rainbow',out_shape = 'CHW'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy().astype(np.float)/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array[:,:,:3]
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        if (tensor.size(0) == 3):
            array = 0.5 + tensor.numpy()*0.5
        elif (tensor.size(0) == 2):
            array = tensor.numpy()

    if out_shape == 'HWC':
        array = array.transpose(1,2,0)
    return array

def tensor2array2(tensor,min=0,max=None):
    if max==None:
        max = tensor.max().item()

    if len(tensor.shape) ==2:
        tensor = tensor.unsqueeze(0)
    arr=  (tensor/max).cpu().numpy()
    return arr


# =============custom
kernel_ver = torch.tensor([0,1,0,
                        0, 1, 0,
                        0, 1, 0]).type(torch.float).reshape([1,1, 3, 3]).cuda()
kernel_hor = torch.tensor([0,0,0,
                        1, 1, 1,
                        0, 0, 0]).type(torch.float).reshape([1,1, 3, 3]).cuda()
weight75 = torch.tensor([0, 0, 0,0,0,
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1,
                           1,1,1,1,1,
                           1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1,
                           0,0,0,0,0]).type(torch.float).reshape([1,1,7,5])


left = torch.tensor([1, 0, -1,
                     2, 0, -2,
                     1, 0, -1]).type(torch.float).reshape([1,1, 3, 3]).cuda()
right = torch.tensor([-1, 0, 1,
                      -2, 0, 2,
                      -1, 0, 1]).type(torch.float).reshape([1,1, 3, 3]).cuda()  #
def dilation(batch,kernel,iteration=1):

    w_sum = kernel.sum()
    batch =batch.type_as(kernel)
    #batch = batch.unsqueeze(dim=1)
    #batch = torch.tensor(batch).type_as(kernel)

    i =0
    ret=0
    while i<iteration:

        ret = F.conv2d(input=batch, weight=kernel, padding=1)
        ret[ret < w_sum] = 0
        ret /= w_sum
        i+=1

    return ret#.squeeze(dim=1)


def erosion(batch,kernel=kernel_ver,iteration=1):


    w_sum = kernel.sum()
    batch =batch.type_as(kernel)
    #batch = batch.unsqueeze(dim=1)
    i = 0
    ret = 0
    while i < iteration:

        ret = F.conv2d(input=batch, weight=kernel, padding=1)
        ret[ret > 0] = 1
        i+=1
    return ret#.squeeze(dim=1)




def rectify(batch):
    batch = batch.unsqueeze(dim=1)  # bchw

    batch_ed = dilation(erosion(batch, kernel_ver, iteration=3), kernel_ver, iteration=3)

    batch_l = F.conv2d(input=batch_ed, weight=left, padding=1)
    batch_l[batch_l > 1] = 1
    batch_l[batch_l < 0] = 0

    batch_r = F.conv2d(input=batch_ed, weight=right, padding=1)
    batch_r[batch_r > 1] = 1
    batch_r[batch_r < 0] = 0

    batch_lr = (batch_l + batch_r)

    batch_lr[batch_lr > 1] = 1
    batch_lr[batch_lr < 0] = 0
    poles = dilation(erosion(batch_lr, kernel_hor), kernel_ver)
    poles[poles > 0] = 1
    ind_mov = poles + batch
    ind_mov[ind_mov > 1] = 1

    return poles.squeeze(dim=1),ind_mov.squeeze(dim=1)