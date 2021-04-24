
import networks
from path import Path
import torch.optim as optim
import torch


def reframe(mode,inputs,frame_sides):

    if mode =='3din':
        colors = torch.cat([inputs["color_aug",frame_sides[0], 0].unsqueeze(dim=2),
                                inputs["color_aug", frame_sides[1], 0].unsqueeze(dim=2),
                                inputs["color_aug",frame_sides[2], 0].unsqueeze(dim=2)],
                               dim=2)

        return colors
    elif mode =='3in':
        colors = torch.cat([inputs["color_aug",frame_sides[0], 0],
                                     inputs["color_aug", frame_sides[0], 0],
                                     inputs["color_aug", frame_sides[0], 0]],
                                    dim=1)
        return colors
    elif mode =='2in':
        color_prior = torch.cat([inputs["color_aug", frame_sides[0], 0],
                               inputs["color_aug", frame_sides[1], 0],
                               ],
                              dim=1)
        color_next = torch.cat([inputs["color_aug", frame_sides[1], 0],
                               inputs["color_aug", frame_sides[2], 0],
                               ],
                              dim=1)
        return color_prior,color_next
    elif mode =='1in':
        return  inputs["color_aug", 0, 0]

def model_init(model_opt):
    # models
    # details
    print("--> model mode:{}".format(model_opt['mode']))
    print("--> framework :{}".format(model_opt['framework']))

    device = model_opt['device']
    models = {}  # dict
    scales = model_opt['scales']
    lr = model_opt['lr']
    scheduler_step_size = model_opt['scheduler_step_size']
    model_mode = model_opt['mode']
    load_paths = model_opt['load_paths']
    optimizer_path = model_opt['optimizer_path']
    framework = model_opt['framework']

    if framework =='spv':
        pass
    elif framework =='shared':
        models["encoder"] = networks.getEncoder(model_mode[0])
        models["depth"] = networks.getDepthDecoder(model_mode[1])
        models["pose"] = networks.getPoseDecoder(model_mode[3])


    elif framework == 'ind':
        models["depth_encoder"] = networks.getEncoder(model_mode[0])
        models["depth"] = networks.getDepthDecoder(model_mode[1])
        if model_mode[2] not in ['3din','3in','2in']:
            models["pose"] = networks.getPoseCNN(model_mode[2])
        else:

            models["pose_encoder"] = networks.getEncoder(model_mode[2])
            models["pose"] = networks.getPoseDecoder(model_mode[3])

        # encoder



    # model device
    for k, v in models.items():
        models[k].to(device)

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