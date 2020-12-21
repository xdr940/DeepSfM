
import torch
from .erodila import rectify
def float8or(t1,t2):

    return ((t1 + t2) > 0).float()

def float8minus(t1,t2):
    pass
def VarMask(erro_maps):
    '''
    var mask
    :param erro_maps:
    :return:
    '''
    rhosvar = erro_maps.var(dim=1, unbiased=False)  # BHW
    rhosvar_flat = rhosvar.flatten(start_dim=1)  # B,H*W
    median, _ = rhosvar_flat.median(dim=1)  # b
    rhosvar_flat /= median.unsqueeze(1)
    delta_var = rhosvar_flat.mean(dim=1).unsqueeze(1)  # B
    # var_mask = (rhosvar_flat > 0.001).reshape_as(map_0)
    var_mask = (rhosvar_flat > delta_var / 100).reshape_as(rhosvar)

    return  (~var_mask).float()
def MeanMask(erro_maps):
    '''
    mean mask
    :param erro_maps:
    :return:
    '''
    rhosmean = erro_maps.mean(dim=1)  # BHW
    rhosmean_flat = rhosmean.flatten(start_dim=1)  # b,h*w
    delta_mean = rhosmean_flat.mean(dim=1).unsqueeze(dim=1)  # b,1
    mean_mask = (rhosmean_flat < 2 * delta_mean).reshape_as(rhosmean)

    #mean rectify
    #mean_mask = rectify(mean_mask)

    return mean_mask.float()
def IdenticalMask(erro_maps):
    #identity_selection = (idxs_0 >= 2).float()  #
    #rhosmean = erro_maps.mean(dim=1)  # BHW
    #rhosmean_flat = rhosmean.flatten(start_dim=1)  # b,h*w
    #delta_mean = rhosmean_flat.mean(dim=1).unsqueeze(dim=1)  # b,1

    map_12, idxs_12 = torch.min(erro_maps[:,:2,:,:], dim=1)
    map_34, idxs_1 = torch.min(erro_maps[:,2:,:,:], dim=1)


    #identity_selection = (map_12<map_34).float()*((rhosmean_flat < 0.5* delta_mean).reshape_as(rhosmean)).float()#白色更多
    identity_selection = (map_12 < map_34).float()

    #rectiry
    is_over_big = identity_selection.sum(dim=1).sum(dim=1) > (0.7 * 192 * 640)#b
    is_normal = ~is_over_big
        # b#如果identical 部分(1, 白色)大于 70%， 说明摄像机静止， 此时取反或者全黑(mask缩小至0)
    need2 = torch.ones_like(identity_selection).transpose(0, 2).cuda()  # bhw -> hwb
    need = is_normal.float()*need2# b*hwb = hwb
    need = need.transpose(0, 2)  # hwb->bhw
    identity_selection = need * identity_selection

    return  1-identity_selection
