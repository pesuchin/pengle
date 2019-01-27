from pengle.dataset.dataset import Dataset, load_dataset
import pandas as pd
import pytest
import mock
from pandas.util.testing import assert_frame_equal
import feather


def test_dataset_must_refer_to_value():
    dataset = Dataset(data=pd.DataFrame([[1, 2, 3], [1, 2, 3]]),
                      target_column='target',
                      target=[4, 4],
                      target_names=[4],
                      filename='sample.csv')
    assert_frame_equal(dataset.data, pd.DataFrame([[1, 2, 3], [1, 2, 3]]))
    assert dataset.target_column == 'target'
    assert dataset.target == [4, 4]
    assert dataset.target_names == [4]
    assert dataset.filename == 'sample.csv'


def test_objective_option_of_load_dataset():
    df_input_for_classification = pd.DataFrame([[1, 2, 'test'], [1, 2, 'test']], columns=['num', 'num2', 'target'])
    df_input_for_regression = pd.DataFrame([[1, 2, 1], [1, 2, 2]], columns=['num', 'num2', 'target'])
    with mock.patch('pengle.dataset.dataset.load_data_and_save_feather') as load_data_and_save_feather:
        with mock.patch('pengle.dataset.dataset.reduce_mem_usage') as reduce_mem_usage:
            load_data_and_save_feather.return_value = df_input_for_classification
            reduce_mem_usage.return_value = df_input_for_classification
            classification_dataset = load_dataset('test.csv', './', 'classification', target='target')

            load_data_and_save_feather.return_value = df_input_for_regression
            reduce_mem_usage.return_value = df_input_for_regression
            regression_dataset = load_dataset('test.csv', './', 'regression', target='target')
    expected_data = pd.DataFrame([[1, 2], [1, 2]], columns=['num', 'num2'])

    assert_frame_equal(classification_dataset.data, expected_data)
    assert classification_dataset.target == ['test', 'test']
    assert classification_dataset.target_names == ['test']
    assert classification_dataset.target_column == 'target'
    assert classification_dataset.filepath == 'test.csv'

    assert_frame_equal(regression_dataset.data, expected_data)
    assert regression_dataset.target == [1, 2]
    assert regression_dataset.target_column == 'target'
    assert regression_dataset.filepath == 'test.csv'


def test_encode_target_option_of_load_dataset():
    df_target_is_str = pd.DataFrame([[1, 2, 'train'],
                                     [1, 2, 'valid'],
                                     [1, 2, 'test']],
                                    columns=['num', 'num2', 'target'])
    df_target_is_int = pd.DataFrame([[1, 2, 1],
                                     [1, 2, 2],
                                     [1, 2, 3]],
                                    columns=['num', 'num2', 'target'])

    with mock.patch('pengle.dataset.dataset.load_data_and_save_feather') as load_data_and_save_feather:
        with mock.patch('pengle.dataset.dataset.reduce_mem_usage') as reduce_mem_usage:
            load_data_and_save_feather.return_value = df_target_is_str
            reduce_mem_usage.return_value = df_target_is_str
            dataset_str = load_dataset('test.csv', './', 'classification',
                                       encode_target=True, target='target')

            load_data_and_save_feather.return_value = df_target_is_int
            reduce_mem_usage.return_value = df_target_is_int
            dataset_int = load_dataset('test.csv', './', 'classification',
                                       encode_target=True, target='target')

    expected_data = pd.DataFrame([[1, 2], [1, 2], [1, 2]], columns=['num', 'num2'])
    assert_frame_equal(dataset_str.data, expected_data)

    assert dataset_str.target == [1, 2, 0]
    assert dataset_str.target_names == ['train', 'valid', 'test']
    assert dataset_str.target_column == 'target'
    assert dataset_str.filepath == 'test.csv'

    assert_frame_equal(dataset_int.data, expected_data)
    assert dataset_int.target == [0, 1, 2]
    assert dataset_int.target_column == 'target'
    assert dataset_int.filepath == 'test.csv'
