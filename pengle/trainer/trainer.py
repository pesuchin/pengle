import pandas as pd
from pengle.transformer.base import Feature, Preprocessor


class FeaturePipeline():
    """特徴抽出から学習を回すためのクラス.
    """

    def __init__(self, steps, save_feature=False):
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

    def run(self, train_dataset, test_dataset):
        all_train = train_dataset.data
        all_test = test_dataset.data
        # TODO: Preprocessorの場合の処理とFeatureの場合の処理を上手く区別する方法を考える
        for feature_class in self.steps:
            if feature_class.__bases__[0] == Preprocessor:
                train_dataset.data, test_dataset.data = feature_class.fit(
                    train_dataset, test_dataset).transform(save=self.save_feature)
            elif feature_class.__bases__[0] == Feature:
                train, test = feature_class.fit(train_dataset, test_dataset).transform(save=self.save_feature)

                all_train = pd.concat([all_train, train], axis=1)
                all_test = pd.concat([all_test, test], axis=1)

        return all_train, all_test
