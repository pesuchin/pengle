
from pandas.util.testing import assert_frame_equal


def assert_classification_dataset_equal(actual_dataset, expected_dataset):
    assert_regression_dataset_equal(actual_dataset, expected_dataset)
    assert actual_dataset.target_names == expected_dataset.target_names


def assert_regression_dataset_equal(actual_dataset, expected_dataset):
    assert_frame_equal(actual_dataset.data, expected_dataset.data)
    assert actual_dataset.target == expected_dataset.target
    assert actual_dataset.target_column == expected_dataset.target_column
    assert actual_dataset.file_path == expected_dataset.file_path
