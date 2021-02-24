
import networks
from path import Path
import torch.optim as optim
import torch

def model_init(model_opt):
    # models
    # details
    print("-> framework mode:{}".format(model_opt['mode']))
    device = model_opt['device']
    models = {}  # dict
    scales = model_opt['scales']
    lr = model_opt['lr']
    scheduler_step_size = model_opt['scheduler_step_size']
    framework_mode = model_opt['mode']
    load_paths = model_opt['load_paths']
    optimizer_path = model_opt['optimizer_path']

    # encoder
    models["encoder"] = networks.getEncoder(framework_mode[0])
    models["depth"] = networks.getDepthDecoder()

    if framework_mode[1] == "fin-2out":
        models["pose"] = networks.getPoseDecoder("fin-2out")
    else:
        models["pose"] = networks.getPoseNet(framework_mode[1])

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
    for name, path in load_paths.items():
        path = load_paths[name]
        if name in models.keys() and path:
            model_dict = models[name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[name].load_state_dict(model_dict)
        print("\t{}:{}".format(name, path))
        # loading adam state

    if optimizer_path:
        optimizer_path = Path(optimizer_path)
        optimizer_dict = torch.load(optimizer_path)
        model_optimizer.load_state_dict(optimizer_dict)
        print('optimizer params from {}'.format(optimizer_path))

    else:
        print('optimizer params from scratch')

    return models, model_optimizer, model_lr_scheduler