"""
@author: sun
@file: set.py
@time: 2018/8/9 13:56
"""
import logging
from logging.handlers import RotatingFileHandler
import os


class Log(object):

    def __init__(self):
        self.base_dir = self.get_base_dir()
        self.logger = self.build_logger('{}{}'.format(self.base_dir, '/log_dir/log.log'))

    @staticmethod
    def get_base_dir():
        base_dir = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
        base_dir = '/'.join(base_dir.split('/')[:-1])
        # base_dir = base_dir.rstrip('/{}'.format(base_dir.split('/')[-1]))
        return base_dir

    @staticmethod
    def build_logger(log_file_name):
        os.makedirs('{}{}'.format(Log.get_base_dir(), '/log_dir'), exist_ok=True)
        # create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            # create file handler
            # log_path = '{}{}'.format(base_dir, '\\data\\redshift_keeper.log_dir')
            log_path = log_file_name
            # fh = logging.FileHandler(log_path)
            fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=2, encoding='utf8')
            fh.setLevel(logging.INFO)
            sh = logging.StreamHandler()

            # create formatter
            fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
            datefmt = "%a %d %b %Y %H:%M:%S"
            formatter = logging.Formatter(fmt, datefmt)

            # add handler and formatter to logger
            fh.setFormatter(formatter)
            sh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger
