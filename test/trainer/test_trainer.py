from pengle.trainer.trainer import Pipeline
from pengle.transformer.categorical_features import *


def test_trainer():
    train_dataset = Dataset(data=pd.DataFrame([[1, 1],
                                               [1, 1],
                                               [1, 2],
                                               [1, 2],
                                               [2, 1]],
                                              columns=['category1', 'category2']),
                            target=[0, 2, 3, 4, 5],
                            target_column='target')

    test_dataset = Dataset(data=pd.DataFrame([[1, 1],
                                              [1, 1],
                                              [1, 2],
                                              [1, 2],
                                              [3, 1],
                                              [2, 1]], columns=['category1', 'category2']),
                           target=[0, 2, 3, 4, 5],
                           target_column='target')
    columns = ['category1', 'category2']
    steps = [BackwardDifferenceEncoder(columns), HelmertEncoder(columns)]
    pipeline = Pipeline(train_dataset, test_dataset, steps).run()
    assert pipeline == steps
