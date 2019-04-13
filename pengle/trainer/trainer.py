import pandas as pd
from logging import getLogger


logger = getLogger(__name__)


class FeaturePipeline():
    """特徴抽出から学習を回すためのクラス.
    """

    def __init__(self, steps, save_feature=False, drop_original_columns=False):
        """
        Arguments:
            steps {list} -- 特徴量操作の一覧

        >>> columns = ['column1', 'column2']
        >>> steps = [
        >>>     BackwardDifferenceEncoder(columns),
        >>>     HelmertEncoder(columns)
        >>> ]
        >>> FeaturePipeline(train_dataset, test_dataset, steps).run()
        """
        self.steps = steps
        self.save_feature = save_feature
        self.drop_original_columns = drop_original_columns

    def run(self, train_dataset, test_dataset):
        all_train = train_dataset.data
        all_test = test_dataset.data
        original_columns_name = list(train_dataset.data.columns)
        for feature_class in self.steps:
            # FIXME: もっとスマートにタスクの区別をする方法があれば修正
            if feature_class.task_name == 'Preprocessor':
                train_dataset, test_dataset = feature_class.fit(
                    train_dataset, test_dataset).transform(save=self.save_feature)
            elif feature_class.task_name == 'Feature':
                train, test = feature_class.fit(
                    train_dataset, test_dataset).transform(save=self.save_feature)

                all_train = pd.concat([all_train, train], axis=1)
                all_test = pd.concat([all_test, test], axis=1)

        if self.drop_original_columns:
            all_train = all_train.drop(original_columns_name, axis=1)
            all_test = all_test.drop(original_columns_name, axis=1)

        return all_train, all_test
