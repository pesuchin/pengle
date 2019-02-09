import re
import time
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd


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
    >>> FamilySize().fit().transform()

    Raises:
        NotImplementedError -- 継承して実装されていない場合にraiseされるエラー

    """

    def __init__(self, prefix='', suffix='', dir='./output/features/'):
        self.name = self.__class__.__name__
        self.dir = dir
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, train_dataset, test_dataset):
        with timer(self.name):
            self.train = pd.DataFrame()
            self.test = pd.DataFrame()
            self.create_features(train_dataset, test_dataset)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = [prefix + column + suffix for column in self.train.columns]
            self.test.columns = [prefix + column + suffix for column in self.test.columns]
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def transform(self, save=False):
        if save:
            self.train.to_feather(str(self.train_path))
            self.test.to_feather(str(self.test_path))
        return self.train, self.test

    def fit_transform(self, train_dataset, test_dataset, save=False):
        self.train, self.test = self.fit(train_dataset, test_dataset).transform(save=save)
        return self.train, self.test


class Preprocessor(metaclass=ABCMeta):
    """前処理クラスのベースになるクラス。継承して使う。

    例)
    >>> class ComplementMissingValue(Preprocessor):
    >>>     def apply(self):
    >>>         for column in self.columns:
    >>>         agg_result = self.agg_func(train_dataset.data[column])
    >>>         train_dataset.data[column].fillna(agg_result, inplace=True)
    >>>         test_dataset.data[column].fillna(agg_result, inplace=True)
    >>> train_dataset, test_dataset = ComplementMissingValue(['Age']).fit(train_dataset, test_dataset).transform()

    Raises:
        NotImplementedError -- 継承して実装されていない場合にraiseされるエラー

    """

    def __init__(self, prefix='', suffix='', dir='./output/features/'):
        self.name = self.__class__.__name__
        self.dir = dir
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, train_dataset, test_dataset):
        with timer(self.name):
            self.train_dataset, self.test_dataset = self.apply(train_dataset, test_dataset)
        return self

    @abstractmethod
    def apply(self):
        raise NotImplementedError

    def transform(self, save=False):
        if save:
            self.train_dataset.to_feather(str(self.train_path))
            self.test_dataset.to_feather(str(self.test_path))
        return self.train_dataset, self.test_dataset

    def fit_transform(self, train_dataset, test_dataset, save=False):
        self.train_dataset, self.test_dataset = self.fit(train_dataset, test_dataset).transform(save=save)
        return self.train_dataset, self.test_dataset
