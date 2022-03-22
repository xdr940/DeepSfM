
import networks
from path import Path
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

from utils.official import readlines
from datasets import KITTIRAWDataset
from datasets import MCDataset
from datasets import CustomMonoDataset

def reframe(component,inputs,frame_sides,key='color_aug'):
    '''

    :param mode: encoder mode
    :param inputs:
    :param frame_sides: training seq input mode
    :param key:
    :return:
    '''
    if component =='3din':
        colors = torch.cat([inputs[key,frame_sides[0], 0].unsqueeze(dim=2),
                                inputs[key, frame_sides[1], 0].unsqueeze(dim=2),
                                inputs[key,frame_sides[2], 0].unsqueeze(dim=2)],
                               dim=2)

        return colors
    elif component =='3in':
        colors = torch.cat([inputs[key,frame_sides[0], 0],
                                     inputs[key, frame_sides[0], 0],
                                     inputs[key, frame_sides[0], 0]],
                                    dim=1)
        return colors
    elif component =='2in':
        color_prior = torch.cat([inputs[key, frame_sides[0], 0],
                               inputs[key, frame_sides[1], 0],
                               ],
                              dim=1)
        color_next = torch.cat([inputs[key, frame_sides[1], 0],
                               inputs[key, frame_sides[2], 0],
                               ],
                              dim=1)
        return color_prior,color_next
    elif component =='1in':
        return  inputs[key, 0, 0]

def model_init(opts):
    # global
    device = opts['device']


    # local
    model_opt = opts['model']
    print("--> components:{}".format(model_opt['components']))
    print("--> paradigm :{}".format(model_opt['paradigm']))

    models = {}  # dict
    lr = model_opt['lr']
    scheduler_step_size = model_opt['scheduler_step_size']
    components = model_opt['components']
    load_paths = model_opt['load_paths']
    optimizer_path = model_opt['optimizer_path']
    paradigm = model_opt['paradigm']

    if paradigm =='spv':
        pass
    elif paradigm =='shared':
        models["encoder"] = networks.getEncoder(components[0])
        models["depth"] = networks.getDepthDecoder(components[1])
        models["pose"] = networks.getPoseDecoder(components[3])


    elif paradigm == 'ind':
        models["depth_encoder"] = networks.getEncoder(components[0])
        models["depth"] = networks.getDepthDecoder(components[1])
        if components[2] not in ['3din','3in','2in']:
            models["pose"] = networks.getPoseCNN(components[2])
        else:

            models["pose_encoder"] = networks.getEncoder(components[2])
            models["pose"] = networks.getPoseDecoder(components[3])

        # encoder

    # 声明所有可用设备
    for name in models.keys():
        models[name] = torch.nn.DataParallel(models[name], device_ids=list(range(len(device))))
    # model device
    for k, v in models.items():
        models[k].to(device[0])

    # params to train
    parameters_to_train = []
    for k, v in models.items():
        parameters_to_train += list(v.parameters())

    model_optimizer = optim.Adam(parameters_to_train, lr)

    model_lr_scheduler = optim.lr_scheduler.StepLR(
        model_optimizer,
        scheduler_step_size,
        lr
    )  # end models arch

    print('--> load models:')

    # load models
    for name,model in models.items():
        if name in list(load_paths.keys()):
            path = load_paths[name]
            if not path:
                continue
            model_dict = models[name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[name].load_state_dict(model_dict)
            print(" ->{}:{}".format(name, path))



    if optimizer_path:
        optimizer_path = Path(optimizer_path)
        optimizer_dict = torch.load(optimizer_path)
        model_optimizer.load_state_dict(optimizer_dict)
        print('optimizer params from {}'.format(optimizer_path))

    else:
        print('optimizer params from scratch')

 

    return models, model_optimizer, model_lr_scheduler


def dataset_init(opts):

    # datasets setting


    #global

    feed_height = opts['feed_height']
    feed_width = opts['feed_width']
    dataset_opt = opts['dataset']
    frame_sides = opts['frame_sides']
    scales = opts['scales']

    device = opts['device']
    # local
    datasets_dict = {"kitti": KITTIRAWDataset,
                     # "kitti_odom": KITTIOdomDataset,
                     "mc": MCDataset,
                     "custom_mono": CustomMonoDataset}

    if dataset_opt['type'] in datasets_dict.keys():
        dataset = datasets_dict[dataset_opt['type']]  # 选择建立哪个类，这里kitti，返回构造函数句柄
    else:
        dataset = CustomMonoDataset

    split_path = Path(dataset_opt['split']['path'])
    train_path = split_path / dataset_opt['split']['train_file']
    val_path = split_path / dataset_opt['split']['val_file']
    data_path = Path(dataset_opt['path'])
    if not data_path.exists():
        print("data path error")
        exit(-1)


    batch_size = dataset_opt['batch_size']
    num_workers = dataset_opt['num_workers']

    train_filenames = readlines(train_path)
    val_filenames = readlines(val_path)
    img_ext = '.png'



    # train loader
    train_dataset = dataset(  # KITTIRAWData
        data_path=data_path,
        filenames=train_filenames,
        height=feed_height,
        width=feed_width,
        frame_sides=frame_sides,  # kitti[0,-1,1],mc[-1,0,1]
        num_scales=len(scales),
        mode="train"
        # img_ext='.png'
    )
    train_loader = DataLoader(  # train_datasets:KITTIRAWDataset
        dataset=train_dataset,
        batch_size=batch_size*len(device),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    # val loader
    val_dataset = dataset(
        data_path=data_path,
        filenames=val_filenames,
        height=feed_height,
        width=feed_width,
        frame_sides=frame_sides,
        num_scales=len(scales),
        mode="val",
        img_ext=img_ext)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size*len(device),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    print("Using split:{}, {}, {}".format(split_path,
                                          dataset_opt['split']['train_file'],
                                          dataset_opt['split']['val_file']
                                          ))
    print("There are {:d} training items and {:d} validation items".format(
        len(train_dataset), len(val_dataset)))

    return train_loader, val_loader
