import argparse

import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--save_dir', type=str, default='../checkpoints/', help='save directory')
    parser.add_argument('--load_dir', type=str, default='../checkpoints/UNet_BraTS2023_epoch_100.pth', help='load directory')
    parser.add_argument('--root_dir', type=str, default='../data/BraTS2023/', help='root directory of the dataset')
    parser.add_argument('--mask', type=str, default='config/UNet_2d_random_mask.yaml', help='config of mask')
    parser.add_argument('--dataset', type=str, default='BraTS2023', help='dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model', type=str, default='UNet', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--loss', type=str, default='DiceLoss', help='loss function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='learning rate scheduler')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)