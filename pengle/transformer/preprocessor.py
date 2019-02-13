import numpy as np
from pengle.dataset.dataset import Dataset
from pengle.transformer.base import Preprocessor, timer
import copy


class ComplementMissingValue(Preprocessor):
    def __init__(self, columns, agg_func=np.mean):
        super().__init__()
        self.columns = columns
        self.agg_func = agg_func

    def apply(self, train_dataset, test_dataset):
        train = copy.deepcopy(train_dataset)
        test = copy.deepcopy(test_dataset)
        for column in self.columns:
            agg_result = self.agg_func(train_dataset.data[column])
            train.data[column].fillna(agg_result, inplace=True)
            test.data[column].fillna(agg_result, inplace=True)
        return train, test


class ExtractStrings(Preprocessor):
    def __init__(self, columns, regexps):
        super().__init__()
        self.columns = columns
        self.regexps = regexps

    def apply(self, train_dataset, test_dataset):
        train = copy.deepcopy(train_dataset)
        test = copy.deepcopy(test_dataset)
        for column, regexp in zip(self.columns, self.regexps):
            train.data[column] = train_dataset.data[column].str.extract(regexp, expand=False)
            test.data[column] = test_dataset.data[column].str.extract(regexp, expand=False)
        return train, test


class ReplaceStrings(Preprocessor):
    def __init__(self, columns, replace_rule):
        super().__init__()
        self.columns = columns
        self.replace_rule = replace_rule

    def apply(self, train_dataset, test_dataset):
        train = copy.deepcopy(train_dataset)
        test = copy.deepcopy(test_dataset)
        for column in self.columns:
            train.data[column].replace(self.replace_rule, inplace=True)
            test.data[column].replace(self.replace_rule, inplace=True)
        return train, test
