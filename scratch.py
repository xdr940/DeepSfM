import numpy as np

gt_depths = np.load("/home/roit/datasets/splits/eigen_std/gt_depths.npz", allow_pickle=True)
gt_depths = gt_depths["data"]
gts = []
cnt=0
for gt in gt_depths:
    gt = np.expand_dims(gt,axis=0)
    gts.append(gt)
    cnt+=1
    if cnt>=10:
        break


gts=np.concatenate(gts,axis=0)


mc_pred = np.load('mc_gt.npy')

of_pred = np.load('of_gt.npy')


print('ok')
