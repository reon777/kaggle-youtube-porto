import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklean.linear_model import LogisticRegression

from load_data import load_train_data, loat_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'

if __name = '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.addHandler(DEBUG)
    logger.addHandler(handler)

    x_train = df.drop('target', axis=1)
    y_train = df['target'].values

    

