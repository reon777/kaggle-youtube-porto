import pandas as pd
import numpy as np
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

if __name__ =='__main__':

    # ------------ログ設定----------------------
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    # 標準出力のログ
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)
    # ログファイル出力のログ
    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)

    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    # ------------前処理----------------------
    logger.info('start')

    df = load_train_data()

    # columns指定なのでaxis=1
    x_train = df.drop('target', axis=1)
    y_train = df['target'].values

    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(x_train.shape))

    # ------------学習----------------------

    # random_stateに数字を指定することで結果を再現できる
    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)
    logger.info('train end')

    # ------------モデル評価----------------------

    df = load_test_data()
    x_test = df[use_cols].sort_values('id')

    logger.info('test data load {}'.format(x_test.shape))

    pred_test = clf.predict_proba(x_test)

    logger.info('test end')

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)
