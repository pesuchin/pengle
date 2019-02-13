from pengle.trainer.trainer import FeaturePipeline
from pengle.transformer.categorical_features import *
from pandas.util.testing import assert_frame_equal


def test_FeaturePipeline():
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
    train, test = FeaturePipeline(steps).run(train_dataset, test_dataset)
    assert_frame_equal(train, pd.DataFrame([[1, 1, 1, -0.5, -0.5, 1, -1.0, -1.0],
                                            [1, 1, 1, -0.5, -0.5, 1, -1.0, -1.0],
                                            [1, 2, 1, -0.5, 0.5, 1, -1.0, 1.0],
                                            [1, 2, 1, -0.5, 0.5, 1, -1.0, 1.0],
                                            [2, 1, 1, 0.5, -0.5, 1, 1.0, -1.0]],
                                           columns=['category1', 'category2', 'intercept_BackwardDifferenceEncoder',
                                                    'category1_0_BackwardDifferenceEncoder', 'category2_0_BackwardDifferenceEncoder',
                                                    'intercept_HelmertEncoder', 'category1_0_HelmertEncoder',
                                                    'category2_0_HelmertEncoder']))
