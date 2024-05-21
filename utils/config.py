import yaml
import argparse


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_args():
    parser = argparse.ArgumentParser(description='Train a deep learning model with specified configuration.')
    parser.add_argument('-c', '--config', type=str, default='config/UNet_2d.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument('-d', '--device', type=str, help='Device to run the model on, e.g., "cuda:0" or "cpu"')
    parser.add_argument('-m', '--model', type=str, help='Model to use for training')
    parser.add_argument('-p', '--pretrain', type=bool, help='Whether to use a pre-trained model')
    parser.add_argument('--load_dir', type=str, help='Directory to load the pre-trained model from')

    return parser.parse_args()
