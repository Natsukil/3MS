import yaml
import argparse


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_args():
    parser = argparse.ArgumentParser(description='Train/Test a deep learning model with specified configuration.')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to the YAML configuration file')
    parser.add_argument('-d', '--device', type=str, help='Device to run the model on, e.g., "cuda:0" or "cpu"')
    parser.add_argument('-m', '--model', type=str, help='Model to use for training')
    parser.add_argument('-p', '--pretrain', type=bool, help='Whether to use a pre-trained model')
    parser.add_argument('--load_dir', type=str, help='Directory to load the pre-trained model from')
    parser.add_argument('-r', '--resume', type=bool, help='Whether to train from a checkpoint')
    parser.add_argument('--resume_root', type=str, help='File to load the resume model from')

    parser.add_argument('--concat', type=str, help='image concat method')
    parser.add_argument("--mask_kernel_size", type=int, help="mask kernel size")
    parser.add_argument("--train_binary_mask", type=str, help="train_binary mask")
    parser.add_argument("--train_mask_rate", type=float, help="train_mask rate")
    parser.add_argument("--test_binary_mask", type=str, help="test_binary mask")
    parser.add_argument("--test_mask_rate", type=float, help="test_mask rate")
    parser.add_argument("--num_works", type=int, help="dataloader num_works")


    return parser.parse_args()
