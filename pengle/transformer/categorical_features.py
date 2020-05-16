import category_encoders as ce
from pengle.transformer.base import Feature, timer
import pandas as pd
import numpy as np


def get_simple_categorycal_features():
    # BUG: HashingEncoderがyの値をnanにしてしまうバグがあるっぽいので、一旦外した
    print('''
    0: BackwardDifferenceEncoder
    1: BinaryEncoder
    2: HelmertEncoder
    3: OneHotEncoder
    4: OrdinalEncoder
    5: SumEncoder
    6: PolynomialEncoder
    7: BaseNEncoder
    8: LeaveOneOutEncoder''')
    return [
        BackwardDifferenceEncoder, BinaryEncoder, HelmertEncoder,
        OneHotEncoder, OrdinalEncoder, SumEncoder, PolynomialEncoder,
        BaseNEncoder, LeaveOneOutEncoder
    ]


def get_target_features():
    return [TargetEncoder, TargetSmoothingNoiseEncoder]


def get_grouping_features():
    return [TargetStatisticsEncoder]


def get_datetime_encoding_features():
    return [DayEncoding, TimeEncoding]


class BackwardDifferenceEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.BackwardDifferenceEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column +
                       '_BackwardDifferenceEncoder'] = encoded_train[column]
            self.test[column +
                      '_BackwardDifferenceEncoder'] = encoded_test[column]


class BinaryEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.BinaryEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_BinaryEncoder'] = encoded_train[column]
            self.test[column + '_BinaryEncoder'] = encoded_test[column]


class HashingEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.HashingEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_HashingEncoder'] = encoded_train[column]
            self.test[column + '_HashingEncoder'] = encoded_test[column]


class HelmertEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.HelmertEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_HelmertEncoder'] = encoded_train[column]
            self.test[column + '_HelmertEncoder'] = encoded_test[column]


class OneHotEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.OneHotEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_OneHotEncoder'] = encoded_train[column]
            self.test[column + '_OneHotEncoder'] = encoded_test[column]


class OrdinalEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.OrdinalEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_OrdinalEncoder'] = encoded_train[column]
            self.test[column + '_OrdinalEncoder'] = encoded_test[column]


class SumEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.SumEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_SumEncoder'] = encoded_train[column]
            self.test[column + '_SumEncoder'] = encoded_test[column]


class PolynomialEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.PolynomialEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_PolynomialEncoder'] = encoded_train[column]
            self.test[column + '_PolynomialEncoder'] = encoded_test[column]


class BaseNEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.BaseNEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_BaseNEncoder'] = encoded_train[column]
            self.test[column + '_BaseNEncoder'] = encoded_test[column]


class LeaveOneOutEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.LeaveOneOutEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_LeaveOneOutEncoder'] = encoded_train[column]
            self.test[column + '_LeaveOneOutEncoder'] = encoded_test[column]


class TargetEncoder(Feature):
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        encoder = ce.TargetEncoder(cols=self.columns)
        encoder.fit(df_train[self.columns],
                    df_train[self.target_column].values.tolist())
        encoded_train = encoder.transform(df_train[self.columns])
        encoded_test = encoder.transform(df_test[self.columns])
        for column in encoded_train.columns:
            self.train[column + '_TargetEncoder'] = encoded_train[column]
            self.test[column + '_TargetEncoder'] = encoded_test[column]


class TargetSmoothingNoiseEncoder(Feature):
    """単純に目的変数の平均値を求めるだけでなく、smoothingやnoiseを考慮したtarget encoding.
    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    """
    def __init__(self,
                 columns,
                 parameters={
                     'min_samples_leaf': 1,
                     'smoothing': 1,
                     'noise_level': 0
                 }):
        super().__init__(columns)
        self.parameters = parameters

    @staticmethod
    def _add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def _target_encode(self,
                       train_series,
                       test_series,
                       target,
                       min_samples_leaf=1,
                       smoothing=1,
                       noise_level=0):
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        train_series : training categorical feature as a pd.Series
        test_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior
        """
        assert len(train_series) == len(target)
        assert train_series.name == test_series.name

        temp = pd.concat([train_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=train_series.name)[target.name].agg(
            ["mean", "count"])
        # Compute smoothing
        smoothing = 1 / \
            (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

        # Apply average function to all target data
        prior = target.mean()

        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * \
            (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)

        # Apply averages to train and test series
        ft_train_series = pd.merge(
            train_series.to_frame(train_series.name),
            averages.reset_index().rename(columns={
                'index': target.name,
                target.name: 'average'
            }),
            on=train_series.name,
            how='left')['average'].rename(train_series.name +
                                          '_mean').fillna(prior)

        # pd.merge does not keep the index so restore it
        ft_train_series.index = train_series.index
        ft_test_series = pd.merge(
            test_series.to_frame(test_series.name),
            averages.reset_index().rename(columns={
                'index': target.name,
                target.name: 'average'
            }),
            on=test_series.name,
            how='left')['average'].rename(train_series.name +
                                          '_mean').fillna(prior)

        # pd.merge does not keep the index so restore it
        ft_test_series.index = test_series.index
        return self._add_noise(ft_train_series, noise_level), self._add_noise(
            ft_test_series, noise_level)

    def create_features(self, df_train, df_test):
        train_target = df_train[self.target_column]
        for column in self.columns:
            feature_name = column + '_TargetSmoothingNoiseEncoder'
            self.train[feature_name], self.test[
                feature_name] = self._target_encode(
                    df_train[column],
                    df_test[column],
                    train_target,
                    min_samples_leaf=self.parameters['min_samples_leaf'],
                    smoothing=self.parameters['smoothing'],
                    noise_level=self.parameters['noise_level'])


class TargetStatisticsEncoder(Feature):
    """指定したcolumnのそれぞれでgroupbyした時の目的変数の基本統計量を算出する特徴量.

    >>> train, test = TargetStatisticsEncoder().fit(
                        df_train, df_test, groupby_key=['a', 'b'], agg_names=['mean']
                      ).transform()
    """
    def __init__(self,
                 columns,
                 groupby_keys,
                 agg_names=['mean', 'max', 'var', 'std', 'median']):
        super().__init__(columns)
        self.groupby_keys = groupby_keys
        self.agg_names = agg_names

    def create_features(self, df_train, df_test):
        new_columns_dict = {}

        # Target Encodingのための計算部分
        for column in self.groupby_keys:
            new_columns_dict[column] = [column]
            train = pd.DataFrame()
            train[column] = df_train[column]
            for agg_name in self.agg_names:
                new_column_name = 'target_enc_' + agg_name + '_' + column
                new_columns_dict[column].append(new_column_name)
                train[new_column_name] = df_train \
                                                      .groupby(column)[df_train[self.target_column]] \
                                                      .transform(agg_name)
            self.train = pd.concat([self.train, train], axis=1)

        # テストデータへの反映部分
        for column in self.groupby_keys:
            # 列をcolumnがgroupbyのキーの時に作成したもののみに絞りつつ、LEFT OUTER JOINするための処理
            test = pd.merge(
                df_test[self.groupby_keys],
                self.train[new_columns_dict[column]].drop_duplicates(
                    subset=column),
                on=column,
                how='left')
            self.test = pd.concat([self.test, test], axis=1)

        # 元々のカテゴリデータの情報は不要のため
        self.train = self.train.drop(self.groupby_keys, axis=1)
        self.test = self.test.drop(self.groupby_keys, axis=1)


class MonthEncoding(Feature):
    """円上に配置することによる月の情報のエンコーディングの処理(Projecting to a circle).

    例)
    >>> train, test = MonthEncoding().fit(
                        df_train, df_test, columns=['a', 'b']
                      ).transform()
    参考: https://qiita.com/shimopino/items/4ef78aa589e43f315113
    """
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        """特徴量生成関数.

        Arguments:
            df_train {Dataset} -- 訓練データセット
            df_test {Dataset} -- テストデータセット
        """
        for column in self.columns:
            series_train = pd.to_datetime(df_train[column])
            series_test = pd.to_datetime(df_test[column])
            self.train[column +
                       '_cos'] = np.cos(2 * np.pi * series_train.dt.month /
                                        series_train.dt.month.max())
            self.train[column +
                       '_sin'] = np.sin(2 * np.pi * series_train.dt.month /
                                        series_train.dt.month.max())
            self.test[column + '_cos'] = np.cos(
                2 * np.pi * series_test.dt.month / series_test.dt.month.max())
            self.test[column + '_sin'] = np.sin(
                2 * np.pi * series_test.dt.month / series_test.dt.month.max())


class DayEncoding(Feature):
    """円上に配置することによる日の情報のエンコーディングの処理(Projecting to a circle).

    例)
    >>> train, test = DayEncoding().fit(
                        df_train, df_test, columns=['a', 'b']
                      ).transform()
    参考: https://qiita.com/shimopino/items/4ef78aa589e43f315113
    """
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        """特徴量生成関数.

        Arguments:
            df_train {Dataset} -- 訓練データセット
            df_test {Dataset} -- テストデータセット
        """
        for column in self.columns:
            series_train = pd.to_datetime(df_train[column])
            series_test = pd.to_datetime(df_test[column])
            self.train[column + '_cos'] = np.cos(
                2 * np.pi * series_train.dt.day / series_train.dt.day.max())
            self.train[column + '_sin'] = np.sin(
                2 * np.pi * series_train.dt.day / series_train.dt.day.max())
            self.test[column + '_cos'] = np.cos(
                2 * np.pi * series_test.dt.day / series_test.dt.day.max())
            self.test[column + '_sin'] = np.sin(
                2 * np.pi * series_test.dt.day / series_test.dt.day.max())


class TimeEncoding(Feature):
    """円上に配置することによる時間の情報のエンコーディングの処理(Projecting to a circle).

    例)
    >>> train, test = TimeEncoding().fit(
                        df_train, df_test, columns=['a', 'b']
                      ).transform()
    参考: https://qiita.com/shimopino/items/4ef78aa589e43f315113
    """
    def __init__(self, columns):
        super().__init__(columns)

    def create_features(self, df_train, df_test):
        """特徴量生成関数.

        Arguments:
            df_train {Dataset} -- 訓練データセット
            df_test {Dataset} -- テストデータセット
        """
        for column in self.columns:
            series_train = pd.to_datetime(df_train[column])
            series_test = pd.to_datetime(df_test[column])
            self.train[column + '_cos'] = np.cos(
                2 * np.pi * series_train.dt.hour / series_train.dt.hour.max())
            self.train[column + '_sin'] = np.sin(
                2 * np.pi * series_train.dt.hour / series_train.dt.hour.max())
            self.test[column + '_cos'] = np.cos(
                2 * np.pi * series_test.dt.hour / series_test.dt.hour.max())
            self.test[column + '_sin'] = np.sin(
                2 * np.pi * series_test.dt.hour / series_test.dt.hour.max())
