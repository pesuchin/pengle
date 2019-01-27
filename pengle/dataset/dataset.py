"""Dataを扱うためのモジュール."""

import os
import pandas as pd
import numpy as np
import feather
from pathlib import Path
from sklearn import preprocessing


class Dataset(dict):
    """Datasetを管理するためのクラスで、辞書型のような使い方ができる.

    また、以下のような方法で値を参照できる
    >>> dataset = Dataset(data='sample')
    >>> dataset.data
    'sample'

    Raises:
        AttributeError -- 存在しない要素にアクセスした時のエラー

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def load_data_and_save_feather(file_path, output_dir_path, dtypes):
    filename, _ = os.path.splitext(os.path.basename(file_path))
    cwd = Path.cwd()
    filename = 'feather_' + filename + '.ftr'
    feather_path = str(cwd / output_dir_path / filename)

    if os.path.exists(feather_path):
        df = feather.read_dataframe(feather_path)
    else:
        df = pd.read_csv(file_path, dtype=dtypes)
        feather.write_dataframe(df, feather_path)
    return df


def load_dataset(file_path, output_dir_path, objective,
                 encode_target=False, target=None, dtypes=None):
    df = load_data_and_save_feather(file_path, output_dir_path, dtypes)
    df = reduce_mem_usage(df, verbose=True)

    dataset = Dataset(filepath=file_path)

    if not target:
        dataset['data'] = df
        return dataset

    if objective == 'classification':
        target_names = df[target].unique().tolist()
        target_values = df[target].values.tolist()
        if encode_target:
            le = preprocessing.LabelEncoder()
            target_values = le.fit_transform(target_values).tolist()
            classes = {target_values[i]: class_name for i, class_name in enumerate(le.classes_)}
            dataset['classes'] = classes

        df = df.drop(target, axis=1)

        dataset['data'] = df
        dataset['target_column'] = target
        dataset['target'] = target_values
        dataset['target_names'] = target_names
        return dataset

    elif objective == 'regression':
        target_values = df[target].values.tolist()
        df = df.drop(target, axis=1)
        dataset['data'] = df
        dataset['target_column'] = target
        dataset['target'] = target_values
        return dataset


def load_features(features, dir_path='./output/features/'):
    """fetherファイルから特徴量を読み込むための関数.

    Arguments:
        features {list} -- 特徴量の名前の一覧

    Keyword Arguments:
        dir_path {str} -- ディレクトリパスがデフォルトから変わっていた時にディレクトリパスを指定する (default: {'./output/features/'})

    Returns:
        X_train {DataFrame} -- 学習用の特徴量
        X_test {DataFrame} -- テスト用の特徴量
    """
    cwd = Path.cwd()
    dir_path = cwd / dir_path
    dfs = [feather.read_dataframe((str(dir_path) + f'/{f}_train.ftr')) for f in features]
    X_train = pd.concat(dfs, axis=1)
    dfs = [feather.read_dataframe(str(dir_path) + f'/{f}_test.ftr') for f in features]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def reduce_mem_usage(df, verbose=True):
    """DataFrameのメモリ使用量を節約するための関数.

    Arguments:
        df {DataFrame} -- 対象のDataFrame

    Keyword Arguments:
        verbose {bool} -- メモリがどれだけ節約できたかを表示 (default: {True})

    Returns:
        [DataFrame] -- メモリ節約後のDataFrame
    """

    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
