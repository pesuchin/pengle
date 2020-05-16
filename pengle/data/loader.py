import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple
import numpy as np

import feather
import pandas as pd
from pengle.data.target import ClassificationConverter


class InterfaceDataLoader(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def convert(self, df: pd.DataFrame, encode_target: bool) -> pd.DataFrame:
        pass


class BaseDataLoader(InterfaceDataLoader):
    def __init__(self, objective_var, target_column, encode_target=False):
        self.encode_target = encode_target
        self.target_column = target_column
        self.objective_var = objective_var

    def convert(self, df):
        if self.objective_var in ['binary_class', 'multi_class']:
            return ClassificationConverter(
                self.target_column,
                encode_target=self.encode_target).convert(df)
        elif self.objective_var == 'regression':
            return df
        else:
            raise NotImplementedError


def reduce_mem_usage(df) -> pd.DataFrame:
    """DataFrameのメモリ使用量を節約するための関数.

    Arguments:
        df {DataFrame} -- 対象のDataFrame

    Returns:
        [DataFrame] -- メモリ節約後のDataFrame
    """

    numerics = [
        'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'
    ]
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type not in numerics:
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            np_int_type_list = [np.int8, np.int16, np.int32, np.int64]
            for np_int_type in np_int_type_list:
                if c_min > np.iinfo(np_int_type).min and c_max < np.iinfo(
                        np_int_type).max:
                    df[col] = df[col].astype(np_int_type)
        else:
            np_float_type_list = [np.float16, np.float32, np.float64]
            for np_float_type in np_float_type_list:
                if c_min > np.finfo(np_float_type).min and c_max < np.finfo(
                        np_float_type).max:
                    df[col] = df[col].astype(np_float_type)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if (start_mem - end_mem) > 0:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def load_local_data(file_path, dtypes) -> pd.DataFrame:
    cwd = Path.cwd()
    output_dir_path = './data'
    base_filename, _ = os.path.splitext(os.path.basename(file_path))
    filename = 'feather_' + base_filename + '.ftr'
    feather_path = str(cwd / output_dir_path / filename)

    if os.path.exists(feather_path):
        df = feather.read_dataframe(feather_path)
    else:
        if dtypes:
            df = pd.read_csv(file_path, dtype=dtypes)
        else:
            df = pd.read_csv(file_path)
    feather.write_dataframe(df, feather_path)
    return df


class CSVLoader(BaseDataLoader):
    def __init__(self,
                 objective_var,
                 target_column,
                 encode_target=False) -> None:
        super().__init__(objective_var, target_column, encode_target)

    def load(self, file_path, dtypes=None):
        df = load_local_data(file_path, dtypes)
        df = reduce_mem_usage(df)
        return self.convert(df)


class InterfaceSplittedDataLoader(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class ReplicationFeatureLoader(InterfaceSplittedDataLoader):
    def __init__(self,
                 objective_var,
                 target_column,
                 feature_name_list,
                 encode_target=False,
                 dir_path='./output/features/'):
        self.dir_path = dir_path
        self.feature_name_list = feature_name_list

    def load(self):
        dir_path = Path.cwd() / self.dir_path
        features = [
            feather.read_dataframe(str(dir_path) + f'/{f}_train.ftr')
            for f in self.feature_name_list
        ]
        df_train = pd.concat(features, axis=1)
        features = [
            feather.read_dataframe(str(dir_path) + f'/{f}_test.ftr')
            for f in self.feature_name_list
        ]
        df_test = pd.concat(features, axis=1)
        df_target_train = feather.read_dataframe(
            str(Path.cwd() / 'data' / 'target_var_train.ftr'))
        df_target_test = feather.read_dataframe(
            str(Path.cwd() / 'data' / 'target_var_test.ftr'))
        df_train = pd.concat([df_train, df_target_train], axis=1)
        df_test = pd.concat([df_test, df_target_test], axis=1)
        return df_train, df_test


class TrainAndTestLoader(InterfaceSplittedDataLoader):
    def __init__(self, loader, train_file_path, test_file_path):
        self.loader = loader
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

    def load(self):
        df_train = self.loader.load(self.train_file_path)
        df_test = self.loader.load(self.test_file_path)
        return df_train, df_test