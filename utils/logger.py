import logging
import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, log_filename):
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.DEBUG)

        # 创建日志目录
        log_dir = os.path.join(log_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 创建文件处理器
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename+".log"))
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def log_config(self, config):
        self.info("Training Configuration:")
        for key, value in config.items():
            self.info(f"{key}: {value}")


class TensorboardLogger:
    def __init__(self, log_dir):
        log_dir = os.path.join(log_dir, "tensorboard_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag, img, step):
        self.writer.add_image(tag, img, step)

    def log_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def close(self):
        self.writer.close()
