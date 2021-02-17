import torch
import torch.nn.functional as F
arr = torch.linspace(1,20,20).reshape([1,1,4,5])
print(arr)

arr_sub = F.interpolate(arr, [3,3], mode="bilinear", align_corners=False)
print(arr_sub)