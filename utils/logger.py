import logging
from pathlib import Path

import colorlog
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, dst='both'):
        self.file_handler = None
        self.console_handler = None

        # 添加处理器到logger
        if dst == 'file':
            self.logger = logging.getLogger('Training_f_Logger')
            self.logger.setLevel(logging.DEBUG)
            self.add_file_handler(log_dir)
            self.logger.addHandler(self.file_handler)
        elif dst == 'console':
            self.logger = logging.getLogger('Training_c_Logger')
            self.logger.setLevel(logging.DEBUG)
            self.add_console_handler()
            self.logger.addHandler(self.console_handler)
        elif dst == 'both':
            self.logger = logging.getLogger('Training_fac_Logger')
            self.logger.setLevel(logging.DEBUG)
            self.add_file_handler(log_dir)
            self.add_console_handler()
            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.console_handler)
        else:
            raise ValueError("dst must be 'file', 'console' or 'both'")

    def add_file_handler(self, log_dir):
        # 创建文件处理器
        self.file_handler = logging.FileHandler(Path(log_dir) / "training.log")
        self.file_handler.setLevel(logging.DEBUG)

        date_fmt = '%m-%d %H:%M:%S'
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt=date_fmt)
        self.file_handler.setFormatter(formatter)

    def add_console_handler(self):
        # 创建控制台处理器
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)

        date_fmt = '%m-%d %H:%M:%S'
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(message)s",
            datefmt=date_fmt,
            log_colors={
                'DEBUG': 'green',
                'INFO': 'cyan',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        self.console_handler.setFormatter(console_formatter)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def enable_console_output(self):
        if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):
            self.logger.addHandler(self.console_handler)

    def disable_console_output(self):
        self.logger.removeHandler(self.console_handler)

    def log_config(self, config):
        self.info("Training Configuration:")
        for key, value in config.items():
            self.info(f"{key}: {value}")


class TensorboardLogger:
    def __init__(self, log_dir):
        # log_dir = os.path.join(log_dir, "tensorboard_logs")
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir, exist_ok=True)
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
