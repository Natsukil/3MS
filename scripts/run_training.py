import sys
import torch
from networks import get_network
from training import ModelInitializer
from training import LossFunctions
from training import OptimizerFactory
from training import SchedulerFactory
from evaluations import evaluate_model
from training import train
from utils import load_config, get_args


def main(args):
    # load config from file
    config = load_config(args.config)

    device = torch.device(
        args.device if args.device else config['train']['device'] if torch.cuda.is_available() else 'cpu')
    model_name = args.model if args.model else config['train']['model']
    pretrain = args.pretrain if args.pretrain is not None else config['train']['pretrain']
    load_dir = args.load_dir if args.load_dir else config['train']['ckpt']

    # get network
    net = get_network(model_name).to(device)

    # load network or init network weights
    if pretrain:
        try:
            net.load_state_dict(torch.load(load_dir))
        except Exception as e:
            print(e)
            print('load pretrain model failed')
            sys.exit(-1)
    else:
        initializer = ModelInitializer(method=config['train']['init_method'], uniform=False)
        initializer.initialize(net)

    # loss function
    criterion = LossFunctions()

    # optimizer
    optimizer_f = OptimizerFactory(config['train']['optimizer'], net, lr=float(config['train']['learning_rate']))

    # learning rate scheduler
    scheduler_f = SchedulerFactory(optimizer_f.optimizer, config['train']['scheduler'])

    # eval
    # metric = MetricFactory
    metric = evaluate_model
    try:
        train(config=config,
              net=net,
              device=device,
              criterion=criterion,
              optimizer=optimizer_f, scheduler=scheduler_f,
              metric=metric,
              )
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    arguments = get_args()
    main(arguments)
