import json
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd
from sklearn import preprocessing


class InterfaceLabelConverter(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, target_column_name: str):
        pass

    @abstractmethod
    def convert(self, dataset, df) -> pd.DataFrame:
        pass


class ClassificationConverter(InterfaceLabelConverter):
    def __init__(self, target_column_name: str, encode_target=False):
        self.target_column_name = target_column_name
        self.encode_target = encode_target

    def convert(self, df):
        if self.encode_target:
            target_values = df[self.target_column_name].values.tolist()
            le = preprocessing.LabelEncoder()
            target_values = le.fit_transform(target_values).tolist()
            classes = {
                target_values[i]: class_name
                for i, class_name in enumerate(le.classes_)
            }
            output_json_path = Path.cwd() / 'data' / 'target_var2class.json'
            with open(output_json_path, 'w') as f:
                json.dump(classes, f, indent=4)
            df[self.target_column_name] = target_values
        return df
