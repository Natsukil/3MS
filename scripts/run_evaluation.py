import sys
from pathlib import Path

import torch
from evaluations import calculate_metrics
from evaluations.eval import evaluation
from networks import get_network
from training import LossFunctions
from utils import load_config, get_args


def main(args):
    # load test config from file
    config = load_config(args.config)

    device = torch.device(
        args.device if args.device else config['test']['device'] if torch.cuda.is_available() else 'cpu')
    model_name = args.model if args.model else config['test']['model']
    ckpt = Path(args.load_dir if args.load_dir else config['test']['ckpt'])

    net = get_network(model_name).to(device)
    # get network weights from file
    if ckpt.exists() and ckpt.is_file():
        net.load_state_dict(torch.load(ckpt))
    else:
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt}")

    # loss function
    criterion = LossFunctions()

    try:
        evaluation(config=config,
                   net=net,
                   device=device,
                   criterion=criterion,
                   show_image=True)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    # test args may need: --config --device --model --load_dir
    arguments = get_args()
    main(arguments)
