from abc import ABCMeta, abstractmethod
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pengle.data.splitter import SequenceSplitter
import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb


class InterfaceTrainer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def set_task_setting(self, target_column, objective_var):
        pass

    @abstractmethod
    def fit(self,
            df_train,
            splitter=SequenceSplitter(test_rate=0.1),
            columns=None,
            categorical_columns=None,
            parameters=None):
        pass

    @abstractmethod
    def predict(self, df_test):
        pass


class BaseTrainer(InterfaceTrainer):
    def set_columns(self, columns):
        self.columns = columns

    def set_task_setting(self, target_column, objective_var):
        self.target_column = target_column
        self.objective_var = objective_var


class RandomForestTrainer(BaseTrainer):
    def __init__(self):
        self.parameters = {
            'n_estimators': 10,
            'criterion': 'gini',
            'max_features': None,
            'max_depth': 15,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0,
            'max_leaf_nodes': None,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 20200222,
            'class_weight': None
        }
        self.columns = None

    def fit(self,
            df_train,
            splitter=SequenceSplitter(test_rate=0.1),
            categorical_columns=None,
            parameters=None):
        if parameters:
            self.parameters = parameters
        X_train = df_train.drop(
            self.target_column,
            axis=1) if self.target_column in df_train.columns else df_train
        X_train = X_train.fillna(-1)
        X_train = X_train[
            self.columns].values if self.columns else X_train.values
        if self.objective_var == 'classification':
            self.model = RandomForestClassifier(**self.parameters)
            self.model.fit(X_train,
                           df_train[self.target_column].values.tolist())
        elif self.objective_var == 'regression':
            # TODO: デフォルトパラメーターを考える
            self.model = RandomForestRegressor()
            self.model.fit(X_train,
                           df_train[self.target_column].values.tolist())

    def predict(self, df_test):
        X_test = df_test.drop(
            self.target_column,
            axis=1) if self.target_column in df_test.columns else df_test
        X_test = X_test.fillna(-1)
        X_test = X_test[self.columns].values if self.columns else X_test.values

        return self.model.predict(X_test), self.model.predict_proba(X_test)


def get_lgbm_default_parameters(objective_var):
    if objective_var == 'regression':
        return {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'l2',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
    elif objective_var == 'binary_class':
        return {
            'learning_rate': 0.03,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'class_weight': 'balanced',
            'num_leaves': 31,
            'verbose': 1,
            'subsample': 0.99,
            'colsample_bytree': 0.99,
            'random_state': 42,
            'max_depth': 15,
            'lambda_l2': 0.02085548700474218,
            'lambda_l1': 0.004107624022751344,
            'bagging_fraction': 0.7934712636944741,
            'feature_fraction': 0.686612409641711,
            'min_child_samples': 21
        }
    elif objective_var == 'multi_class':
        return {
            'learning_rate': 0.03,
            'objective': 'multiclass',
            'class_weight': 'balanced',
            'num_leaves': 31,
            'verbose': 1,
            'subsample': 0.99,
            'colsample_bytree': 0.99,
            'random_state': 42,
            'max_depth': 15,
            'lambda_l2': 0.02085548700474218,
            'lambda_l1': 0.004107624022751344,
            'bagging_fraction': 0.7934712636944741,
            'feature_fraction': 0.686612409641711,
            'min_child_samples': 21
        }


class LGBMTrainer(BaseTrainer):
    def __init__(self, use_optuna=False, metric=None):
        self.columns = None
        self.use_optuna = use_optuna
        self.metric = metric

    def fit(self,
            df_train,
            splitter=SequenceSplitter(test_rate=0.1),
            categorical_columns=[],
            parameters=None):
        if parameters is None:
            self.parameters = get_lgbm_default_parameters(self.objective_var)
        if self.metric is None:
            if self.objective_var == 'binary_class':
                self.parameters['metric'] = 'binary_logloss'
            elif self.objective_var == 'multi_class':
                self.parameters['metric'] = 'multi_logloss'
            elif self.objective_var == 'regression':
                self.parameters['metric'] = 'l2'
        else:
            self.parameters['metric'] = self.metric

        df_splitted_train, df_valid = splitter.split(df_train)
        X_train = df_splitted_train.drop(
            self.target_column, axis=1
        ) if self.target_column in df_splitted_train.columns else df_splitted_train
        X_valid = df_valid.drop(
            self.target_column,
            axis=1) if self.target_column in df_valid.columns else df_valid
        X_train = X_train.fillna(-1)
        X_valid = X_valid.fillna(-1)

        train_data = lgb.Dataset(
            X_train,
            label=df_splitted_train[self.target_column].values.tolist(),
            categorical_feature=categorical_columns)

        val_data = lgb.Dataset(
            X_valid,
            label=df_valid[self.target_column].values.tolist(),
            categorical_feature=categorical_columns)
        if self.use_optuna:
            self.model = optuna_lgb.train(self.parameters,
                                          train_data,
                                          valid_sets=val_data,
                                          verbose_eval=0)
        else:
            self.model = lgb.train(self.parameters,
                                   train_data,
                                   valid_sets=val_data,
                                   verbose_eval=100,
                                   num_boost_round=10000,
                                   early_stopping_rounds=50)

    def predict(self, df_test):
        X_test = df_test.drop(
            self.target_column,
            axis=1) if self.target_column in df_test.columns else df_test
        X_test = X_test.fillna(-1)
        X_test = X_test[self.columns].values if self.columns else X_test.values
        predict_proba = self.model.predict(
            X_test, num_iteration=self.model.best_iteration)
        predictions = [
            1 if predict_proba[i] >= 0.5 else 0
            for i in range(len(predict_proba))
        ]
        return predictions, predict_proba
