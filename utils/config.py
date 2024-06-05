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

    parser.add_argument('-p', '--pretrain', action='store_true', help='Whether to use a pre-trained model')
    parser.add_argument('-np', '--no-pretrain', action='store_false', dest='pretrain',
                        help='Do not use a pre-trained model')

    parser.add_argument('--load_dir', type=str, help='Directory to load the pre-trained model from')
    # 用于设置 resume 为 True 的参数
    parser.add_argument('-r', '--resume', action='store_true', help='Whether to train from a checkpoint')
    # 用于设置 resume 为 False 的参数（如果需要）
    parser.add_argument('--no-resume', action='store_false', dest='resume', help='Do not train from a checkpoint')
    parser.add_argument('--resume_root', type=str, help='File to load the resume model from')

    parser.add_argument('--concat', type=str, help='image concat method')

    parser.add_argument('--learning_rate', type=float, help='learning rate')

    parser.add_argument('--scheduler', type=str, help='scheme')

    # 用于设置 mask_random 为 True 的参数
    parser.add_argument('--mask_random', action='store_true', help='Set mask_random to True')
    # 用于设置 mask_random 为 False 的参数
    parser.add_argument('--no-mask_random', action='store_false', dest='mask_random', help='Set mask_random to False')

    parser.add_argument("--mask_kernel_size", type=int, help="mask kernel size")

    parser.add_argument("--num_works", type=int, help="dataloader num_works")

    parser.add_argument("-dsp", "--description", type=str, default="", help="exp description")

    return parser.parse_args()
