import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import category_encoders as ce
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


# 参考: https://amalog.hateblo.jp/entry/kaggle-feature-management
@contextmanager
def timer(name):
    """時間を測るためのタイマー関数

    Arguments:
        name {str} -- クラス名
    """

    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    """特徴量クラスのベースになるクラス。継承して使う。

    例)
    >>> class FamilySize(Feature):
    >>>     def create_features(self):
    >>>         self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
    >>>         self.test['family_size'] = test['SibSp'] + test['Parch'] + 1
    >>> FamilySize().run().save()

    Raises:
        NotImplementedError -- 継承して実装されていない場合にraiseされるエラー

    """

    def __init__(self, prefix='', suffix='', dir='./output/features/'):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.dir = dir
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'
        self.prefix = prefix
        self.suffix = suffix

    def run(self, train_dataset, test_dataset, columns=[]):
        with timer(self.name):
            self.create_features(train_dataset, test_dataset, columns=columns)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = [prefix + column + suffix for column in self.train.columns]
            self.test.columns = [prefix + column + suffix for column in self.test.columns]
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))
        return self.train, self.test


class BackwardDifferenceEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.BackwardDifferenceEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_BackwardDifferenceEncoder'] = encoded_train[column]
            self.test[column + '_BackwardDifferenceEncoder'] = encoded_test[column]


class BinaryEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.BinaryEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_BinaryEncoder'] = encoded_train[column]
            self.test[column + '_BinaryEncoder'] = encoded_test[column]


class HashingEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.HashingEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_HashingEncoder'] = encoded_train[column]
            self.test[column + '_HashingEncoder'] = encoded_test[column]


class HelmertEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.HelmertEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_HelmertEncoder'] = encoded_train[column]
            self.test[column + '_HelmertEncoder'] = encoded_test[column]


class OneHotEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.OneHotEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_OneHotEncoder'] = encoded_train[column]
            self.test[column + '_OneHotEncoder'] = encoded_test[column]


class OrdinalEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.OrdinalEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_OrdinalEncoder'] = encoded_train[column]
            self.test[column + '_OrdinalEncoder'] = encoded_test[column]


class SumEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.SumEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_SumEncoder'] = encoded_train[column]
            self.test[column + '_SumEncoder'] = encoded_test[column]


class PolynomialEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.PolynomialEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_PolynomialEncoder'] = encoded_train[column]
            self.test[column + '_PolynomialEncoder'] = encoded_test[column]


class BaseNEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.BaseNEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_BaseNEncoder'] = encoded_train[column]
            self.test[column + '_BaseNEncoder'] = encoded_test[column]


class TargetEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.TargetEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_TargetEncoder'] = encoded_train[column]
            self.test[column + '_TargetEncoder'] = encoded_test[column]


class TargetStatisticsEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, groupby_keys, agg_names):
        new_columns_dict = {}
        # Dataset型のdataにはtargetのカラムを入れていないため
        train_dataset.data[train_dataset.target_column] = train_dataset.target
        # Target Encodingのための計算部分
        for column in groupby_keys:
            new_columns_dict[column] = [column]
            train = pd.DataFrame()
            train[column] = train_dataset.data[column]
            for agg_name in agg_names:
                new_column_name = 'target_enc_' + agg_name + '_' + column
                new_columns_dict[column].append(new_column_name)
                train[new_column_name] = train_dataset.data \
                                                      .groupby(column)[train_dataset.target_column] \
                                                      .transform(agg_name)
            self.train = pd.concat([self.train, train], axis=1)

        # テストデータへの反映部分
        for column in groupby_keys:
            # 列をcolumnがgroupbyのキーの時に作成したもののみに絞りつつ、LEFT OUTER JOINするための処理
            test = pd.merge(test_dataset.data[groupby_keys],
                            self.train[new_columns_dict[column]].drop_duplicates(subset=column),
                            on=column, how='left')
            self.test = pd.concat([self.test, test], axis=1)

        # 元々のカテゴリデータの情報は不要のため
        self.train = self.train.drop(groupby_keys, axis=1)
        self.test = self.test.drop(groupby_keys, axis=1)

    def run(self, train_dataset, test_dataset, groupby_keys,
            agg_names=['mean', 'max', 'var', 'std', 'median']):
        with timer(self.name):
            self.create_features(train_dataset, test_dataset, groupby_keys, agg_names)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = [prefix + column + suffix for column in self.train.columns]
            self.test.columns = [prefix + column + suffix for column in self.test.columns]
        return self

class LeaveOneOutEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.LeaveOneOutEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_LeaveOneOutEncoder'] = encoded_train[column]
            self.test[column + '_LeaveOneOutEncoder'] = encoded_test[column]


class MonthEncoding(Feature):
    """円上に配置することによる月の情報のエンコーディングの処理(Projecting to a circle).

    例)
    >>> train, test = MonthEncoding().run(
                        train_dataset, test_dataset, columns=['a', 'b']
                      ).save()
    参考: https://qiita.com/shimopino/items/4ef78aa589e43f315113
    """

    def create_features(self, train_dataset, test_dataset, columns):
        """特徴量生成関数.

        Arguments:
            train_dataset {Dataset} -- 訓練データセット
            test_dataset {Dataset} -- テストデータセット
            columns {list} -- 特徴抽出を行う対象のクラス
        """
        for column in columns:
            series_train = pd.to_datetime(train_dataset.data[column])
            series_test = pd.to_datetime(test_dataset.data[column])
            self.train[column + '_cos'] = np.cos(2 * np.pi * series_train.dt.month / series_train.dt.month.max())
            self.train[column + '_sin'] = np.sin(2 * np.pi * series_train.dt.month / series_train.dt.month.max())
            self.test[column + '_cos'] = np.cos(2 * np.pi * series_test.month.day / series_test.dt.month.max())
            self.test[column + '_sin'] = np.sin(2 * np.pi * series_test.month.day / series_test.dt.month.max())


class DayEncoding(Feature):
    """円上に配置することによる日の情報のエンコーディングの処理(Projecting to a circle).

    例)
    >>> train, test = DayEncoding().run(
                        train_dataset, test_dataset, columns=['a', 'b']
                      ).save()
    参考: https://qiita.com/shimopino/items/4ef78aa589e43f315113
    """

    def create_features(self, train_dataset, test_dataset, columns):
        """特徴量生成関数.

        Arguments:
            train_dataset {Dataset} -- 訓練データセット
            test_dataset {Dataset} -- テストデータセット
            columns {list} -- 特徴抽出を行う対象のクラス
        """
        for column in columns:
            series_train = pd.to_datetime(train_dataset.data[column])
            series_test = pd.to_datetime(test_dataset.data[column])
            self.train[column + '_cos'] = np.cos(2 * np.pi * series_train.dt.day / series_train.dt.day.max())
            self.train[column + '_sin'] = np.sin(2 * np.pi * series_train.dt.day / series_train.dt.day.max())
            self.test[column + '_cos'] = np.cos(2 * np.pi * series_test.dt.day / series_test.dt.day.max())
            self.test[column + '_sin'] = np.sin(2 * np.pi * series_test.dt.day / series_test.dt.day.max())


class TimeEncoding(Feature):
    """円上に配置することによる時間の情報のエンコーディングの処理(Projecting to a circle).

    例)
    >>> train, test = TimeEncoding().run(
                        train_dataset, test_dataset, columns=['a', 'b']
                      ).save()
    参考: https://qiita.com/shimopino/items/4ef78aa589e43f315113
    """

    def create_features(self, train_dataset, test_dataset, columns):
        """特徴量生成関数.

        Arguments:
            train_dataset {Dataset} -- 訓練データセット
            test_dataset {Dataset} -- テストデータセット
            columns {list} -- 特徴抽出を行う対象のクラス
        """
        for column in columns:
            series_train = pd.to_datetime(train_dataset.data[column])
            series_test = pd.to_datetime(test_dataset.data[column])
            self.train[column + '_cos'] = np.cos(2 * np.pi * series_train.dt.hour / series_train.dt.hour.max())
            self.train[column + '_sin'] = np.sin(2 * np.pi * series_train.dt.hour / series_train.dt.hour.max())
            self.test[column + '_cos'] = np.cos(2 * np.pi * series_test.dt.hour / series_test.dt.hour.max())
            self.test[column + '_sin'] = np.sin(2 * np.pi * series_test.dt.hour / series_test.dt.hour.max())


class FundamentalStatistics(Feature):
    """groupbyからの基本統計量による特徴量生成用のクラス.

    例)
    >>> train, test = FundamentalStatistics().run(
                        train_dataset, test_dataset, groupby_keys=['id'], agg_columns=['a', 'b']
                      ).save()
    """

    def create_features(self, train_dataset, test_dataset, groupby_keys, agg_columns):
        """特徴量生成関数.

        Arguments:
            train_dataset {Dataset} -- 訓練データセット
            test_dataset {Dataset} -- テストデータセット
            columns {list} -- 特徴抽出を行う対象のクラス
        """
        agg_funcs = ["min", "max", "sum", "var", "std", "mean", "count", "median"]
        grouping_funcs = {column: agg_funcs for column in agg_columns}
        train = train_dataset.groupby(groupby_keys).agg(agg_columns)
        test = test_dataset.groupby(groupby_keys).agg(agg_columns)
        for column in train.columns:
            self.train[column] = train[column]
            self.test[column] = test[column]
