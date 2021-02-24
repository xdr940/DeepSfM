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
|0224_1333 |[3din,3din]	   |sildurs-h-2k|
|x		   |[3din,3din]	   |sildurs-10k	|