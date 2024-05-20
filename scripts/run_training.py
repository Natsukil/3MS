import sys

import torch
from networks.get_network import get_network
from training.init_weight import  ModelInitializer
from training.loss_functions import LossFunctions
from training.schedulers import SchedulerFactory
from training.train import train
import os
from utils.config import load_config, get_args


def main(args):
    # load config from file
    config = load_config(args.config)

    device = torch.device(
        args.device if args.device else config['train']['device'] if torch.cuda.is_available() else 'cpu')
    model_name = args.model if args.model else config['train']['model']
    pretrain = args.pretrain if args.pretrain is not None else config['train']['pretrain']
    load_dir = args.load_dir if args.load_dir else config['train']['load_dir']

    # get network
    net = get_network(model_name).to(device)

    # load network or init network weights
    if pretrain:
        try:
            net.load_state_dict(torch.load(load_dir))
        except Exception as e:
            print(e)
            print('load pretrain model failed')
    else:
        initializer = ModelInitializer(method=config['train']['init_method'], uniform=False)
        initializer.initialize(net)

    # loss function
    criterion = LossFunctions()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config['train']['lr'])

    # learning rate scheduler
    scheduler = SchedulerFactory.get_scheduler(config['train']['scheduler'], optimizer)

    try:
        train(config, net, device, criterion, optimizer, scheduler)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    arguments = get_args()
    main(arguments)
