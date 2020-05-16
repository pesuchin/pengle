from abc import ABCMeta, abstractmethod
from typing import Tuple
import pandas as pd


class InterfaceSplitter(metaclass=ABCMeta):
    def __init__(self, column: str) -> None:
        pass

    @abstractmethod
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class SequenceSplitter(InterfaceSplitter):
    """データの順番で分割する
    """
    def __init__(self, test_rate=0.1):
        self.test_rate = test_rate

    def split(self, df):
        train_length = int(len(df) * (1 - self.test_rate))
        df_train = df.iloc[:train_length]
        df_test = df.iloc[train_length:]
        return df_train, df_test


class TimeSeriesSplitter(InterfaceSplitter):
    """時系列順でデータを分割する
    """
    def __init__(self, column: str, test_rate=0.1, ascending=True):
        self.time_column = column
        self.test_rate = test_rate
        self.ascending = ascending

    def split(self, df):
        train_length = int(len(df) * (1 - self.test_rate))
        df_train = df.sort_values(self.time_column,
                                  ascending=self.ascending).iloc[:train_length]
        df_test = df.sort_values(self.time_column,
                                 ascending=self.ascending).iloc[train_length:]
        return df_train, df_test


class AttributeSplitter(InterfaceSplitter):
    """Userで分割するなどの属性を利用した分割
    """
    def __init__(self, column: str, test_rate=0.1):
        self.column = column
        self.test_rate = test_rate

    def split(self, df):
        raise NotImplementedError


class RandomSplitter(InterfaceSplitter):
    """ランダムに分割する
    """
    def __init__(self, test_rate=0.1):
        self.test_rate = test_rate

    def split(self, df):
        df_test = df.sample(frac=self.test_rate)
        test_index = list(df_test.index)
        df_train = df.query('index not in @test_index')
        return df_train, df_test
