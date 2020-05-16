# pengle

pengle is a library for efficient modeling and experimentation in data analysis competitions such as kaggle.

# Requirement

- python = "^3.7"
- mlflow = "^1.8.0"
- feather-format = "^0.4.1"
- sklearn = "^0.0"
- category_encoders = "^2.2.2"
- lightgbm = "^2.3.1"

# Installation

You can prepare the environment by simply executing the following commands.

```bash
poetry install
```

# Usage

Place the data you want to analyze under the data directory, and if you have two files, train.csv and test.csv, you can use them as follows.

```python
from pengle.data.loader import CSVLoader, TrainAndTestLoader
from pengle.transformer.categorical_features import *
from pengle.trainer.pipeline import ClassificationPipeline
from pengle.trainer.trainer import LGBMTrainer

columns = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
target_column = 'Survived'
objective = 'binary_class'

loader = CSVLoader(objective, target_column)
df_train, df_test = TrainAndTestLoader(loader, './data/train.csv', './data/test.csv').load()

feat_pipe = BinaryClassPipeline(df_train, df_test, target_column,
                                splitter=RandomSplitter(test_rate=0.1),
                                trainer=LGBMTrainer(),
                                save_feature=True, drop_columns=columns)
# Store an instance of the feature class in a list.
pipes = [
    BackwardDifferenceEncoder(columns),
    HelmertEncoder(columns)
]
feat_pipe.add_pipe(pipes)

# The result of the inference is stored in pred_target.
pred_target = feat_pipe.run()
```

# License

"pengle" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
