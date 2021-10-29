# DeepSfM

DeepSfM是一个关于深度估计网络模型的训练验证、测试框架，源码来自monodepth2改进。
- 网络框架包含有tensorboard，终端进度条等功能,可以定期保存模型到logdir文件夹。
- 另外,还通过.yaml文件来更改配置，灵活调整, 并在训练时将配置文件保存到logdir。
- 框架可以具有统一输入输出结构,即`inputs, outputs`来承载复杂的模型IO以及中间变量.
- 可以多GPU训练.
- 多种范式, 多种单元结合.


DeepSfM is a depth estimation code-framework of training, validation and test, which based on monodepth2.
It attached with several function such as tensorboard, terminal progress bar, logdir etc.



## 文件夹介绍
---

- total
```
./
	|-datasets
	|-networks
	|-scripts
	|-test
	|-utils
```

- 	datasets

主要是关于kitti\mn\custom-mono等数据集的载入,以及dataloader
```bash

data.Dataset ->
	mono_dataset_v2/MonoDataset - > 
		custom_mono/CustomMono
		kitti_dataset_v2/KITTIRAWDataset
		mc_dataset/MCDataset


```
- networks //框架用到的模型
  
  这里描述下layers的嵌套情况

  nn.xxPad
  nn.conv2d
	->conv3x3
	->nn.ELU
		->convBlock
			->
-  scripts //框架涉及到的训练,验证, 评估, 推断等

-  data_prep_scripts//关于数据单独处理部分的脚本







## I/O data structure

---

通过tensor数据的统一管理, 减少模糊与歧义

```apex

inputs:# 33 dict
	#K  4scale
	K,0
	K,1
	K,2
	K,3
	#
	inv_K,0
	inv_k,1
	inv_k,2
	inv_k,3

	color 3side x 4 scale = 12
	aug_color 3side x 4 scale = 12
	depth_gt

outputs #34 dict
	cam2cam 2side
	depth 4 scale
	sample 2side x 4 scale
	disp 4 scale
	color 2x4 #sides pred
	color_identity 2 x4
```
	
	
	


## 部分参数说明

**1. paradigm**
模型包括多种训练范式, 如下

 - shared //共用编码器方式
 - ind	//各自任务独立编码器
 - spv	//有监督模式
 - rebuild	//编码解码同一映射

**2. components**


 __Encoder(DepthEncoder)__ 
 
 components[0]
 
- 1in	//常规的resnet18, 图像输入
- 3in	//层前面改成直接concat, 三帧输入
- 3din	//层前面扩充通道, 三帧输入

__DepthDecoder__
 
 components[1]

- None

__PoseEncoder__

components[2]

- 3din
- c3d

__PoseDecoder__
 components[3]

- fin-2out 
- fin-1out



	
	


## evaluation pipline
---


```apex
evaluate_depth.py

#step 0
	output = depth_decoder(encoder(input_color))
#step 1
	pred_disp, pred_depth_tmp = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
#step 2

 	pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))#1271x341 t0 128x640
    pred_depth = 1 / pred_disp# 也可以根据上面直接得到

#step 3
	pred_depth = pred_depth[mask]#并reshape成1d
	gt_depth = gt_depth[mask]

#step 4
	 if opt.median_scaling:
		ratio = np.median(gt_depth) / np.median(pred_depth)#中位数， 在eval的时候， 将pred值线性变化，尽量能使与gt接近即可
		ratios.append(ratio)
		pred_depth *= ratio

#step 5
 	pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH#所有历史数据中最小的depth, 更新,
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH#...
#step 6
    metric = compute_errors(gt_depth, pred_depth)
    
    
    

trainer.py
#step 0 batch_process
	disp = self.models["depth"](*features)
	outputs[("disp", 0, 0)] = disp[0]
	outputs[("disp", 0, 1)] = disp[1]
	outputs[("disp", 0, 2)] = disp[2]
	outputs[("disp", 0, 3)] = disp[3]

#step 1	generate_img_pred
	disp = outputs[("disp", 0, scale)]
 	disp = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)

	_, depth = disp_to_depth(disp)
    #depth = disp2depth(disp)
    #outputs[("depth", 0, scale)] = depth


# compute_metrics
 depth_pred_full = F.interpolate(outputs[("depth", 0, 0)], [full_height, full_width], mode="bilinear", align_corners=False)
        depth_pred = torch.clamp(
              depth_pred_full,
              min=min_depth,
              max=max_depth
          )

mask = depth_gt > depth_gt.min()
mask*=depth_gt< depth_gt.max()


depth_gt = depth_gt[mask]
depth_pred = depth_pred[mask]


depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)#截断
depth_gt = torch.clamp(depth_gt,min=min_depth,max = max_depth)

metrics = compute_depth_errors(depth_gt, depth_pred)
    

```