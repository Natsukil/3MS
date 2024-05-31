import sys
import torch
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from networks import get_network
from training import ModelInitializer
from training import LossFunctions
from training import OptimizerFactory
from training import SchedulerFactory
from evaluations import calculate_metrics
from training import train
from utils import load_config, get_args, load_checkpoint


def main(args):
    # load config from file
    config = load_config(args.config)

    device = torch.device(
        args.device if args.device else config['train']['device'] if torch.cuda.is_available() else 'cpu')
    model_name = args.model if args.model else config['train']['model']
    pretrain = args.pretrain if args.pretrain is not None else config['train']['pretrain']
    load_dir = args.load_dir if args.load_dir else config['train']['ckpt']

    concat = args.concat if args.concat else config['data']['concat']

    # get network
    net = get_network(model_name, concat).to(device)

    # loss function
    criterion = LossFunctions(concat)

    # optimizer
    optimizer_f = OptimizerFactory(config['train']['optimizer'], net, lr=float(config['train']['learning_rate']))

    # learning rate scheduler
    scheduler_f = SchedulerFactory(optimizer_f.optimizer, config['train']['scheduler'])

    # eval
    # metric = MetricFactory
    metric = calculate_metrics

    # 是否从继续训练
    if args.resume:
        try:
            last_epoch, best_loss, lr = load_checkpoint(args.resume_root, net, optimizer_f, scheduler_f, method='resume')
            config['train']['last_epoch'] = last_epoch
            config['train']['best_loss'] = best_loss
            config['train']['last_learning_rate'] = lr
        except Exception as e:
            print(e)
            print('load from ckpt failed')
            sys.exit(-1)
    # 是否使用预训练模型
    elif pretrain:
        try:
            net.load_state_dict(torch.load(load_dir))
        except Exception as e:
            print(e)
            print('load pretrain model failed')
            sys.exit(-1)
    # 初始化模型
    else:
        initializer = ModelInitializer(method=config['train']['init_method'], uniform=True)
        initializer.initialize(net)

    try:
        train(config=config,
              net=net,
              device=device,
              criterion=criterion,
              optimizer_f=optimizer_f, scheduler_f=scheduler_f,
              metric=metric, resume=args.resume,
              concat_method=concat
              )
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    arguments = get_args()
    main(arguments)
