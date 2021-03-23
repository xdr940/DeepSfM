### framework modes

 __Encoder__ 
- 1in
- 3in
- 3din

__DepthDecoder__
- None

__PoseDecoder__
- fin-2out

__PoseNet__
- 3in
- 3din 
	
	
### DeepSfMLearner 实验记录



|  date   |framework|dataset | metrics |
|  ----  | ---|----  |---|
|0222_2335 |[3din,3in]     |sildurs-2k 	| &   0.211  &   0.845  &   5.068  &   0.256  &   0.777  &   0.912  &   0.952  \\ |
|0223_1054 |[3din,fin-2out]|sildurs-2k	| 0.231  &   1.105  &   5.667  &   0.277  &   0.748  &   0.899  &   0.949  \\|
| x		   |[3in,3in] 	   |sildurs-2k	|
|x 		   |[3in,fin-2out] |sildurs-2k	|
|0223_0038 |[1in,3din]	   |sildurs-2k  |&   0.248  &   1.305  &   4.929  &   0.277  &   0.758  &   0.899  &   0.946  \\|
|0223_0956 |[1in,3in]      |sildurs-2k 	|&   0.245  &   1.207  &   5.431  &   0.283  &   0.746  &   0.894  &   0.942  \\|
|0223_2024 |[3din,3din]	   |sildurs-2k	|&   0.229  &   1.323  &   5.078  &   0.262  &   0.786  &   0.908  &   0.952  \\|
|x		   |[3din,fin-2out]|			|	
|0224_1124 |[1in,3din] 	   |sildurs-h-2k|&   0.253  &   1.370  &   5.442  &   0.284  &   0.765  &   0.890  &   0.938  \\|
|0224_1230 |[3din,3din]    |sildurs-2k	|&   0.228  &   1.407  &   4.914  &   0.254  &   0.786  &   0.917  &   0.952  \\|
|0224_1333 |[3din,3din]	   |sildurs-h-2k|&   0.261  &   2.197  &   4.885  &   0.271  &   0.762  &   0.909  &   0.951  \\
|x		   |[3din,3din]	   |sildurs-10k	|



### evaluation pipline



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