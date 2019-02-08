
class Pipeline():
    """特徴抽出から学習を回すためのクラス.
    """

    def __init__(self, train_dataset, test_dataset, steps):
        """
        Arguments:
            steps {list} -- 特徴量操作の一覧

        >>> steps = [
        >>>     BackwardDifferenceEncoder(),
        >>>     HelmertEncoder()
        >>> ]
        """
        self.steps = steps

    def run(self):
        for feature_class in self.steps:
            pass
        return self.steps
