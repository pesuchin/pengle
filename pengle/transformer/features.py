import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import category_encoders as ce
import pandas as pd


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    """Feature Base Class.

    Example:
    >>> class FamilySize(Feature):
    >>>     def create_features(self):
    >>>         self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
    >>>         self.test['family_size'] = test['SibSp'] + test['Parch'] + 1
    >>> FamilySize().run().save()

    Raises:
        NotImplementedError -- 継承して実装されていない場合にraiseされるエラー

    Returns:
        [type] -- [description]

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


class LeaveOneOutEncoder(Feature):
    def create_features(self, train_dataset, test_dataset, columns):
        encoder = ce.LeaveOneOutEncoder(cols=columns)
        encoder.fit(train_dataset.data[columns], train_dataset.target)
        encoded_train = encoder.transform(train_dataset.data[columns])
        encoded_test = encoder.transform(test_dataset.data[columns])
        for column in encoded_train.columns:
            self.train[column + '_LeaveOneOutEncoder'] = encoded_train[column]
            self.test[column + '_LeaveOneOutEncoder'] = encoded_test[column]