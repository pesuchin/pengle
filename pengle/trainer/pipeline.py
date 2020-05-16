from abc import ABCMeta, abstractmethod

import feather
import pandas as pd
from pathlib import Path
from pengle.data.splitter import SequenceSplitter
from pengle.evaluate.evaluator import Evaluator
from pengle.evaluate.reporter import Reporter
from pengle.trainer.trainer import RandomForestTrainer


class InterfacePipeLine(metaclass=ABCMeta):
    def __init__(self,
                 df_train,
                 df_test,
                 target_column,
                 splitter=SequenceSplitter(test_rate=0.1),
                 trainer=RandomForestTrainer(),
                 drop_columns=[],
                 save_feature=False):
        pass

    @abstractmethod
    def run(self, df):
        pass


class BinaryClassPipeline(InterfacePipeLine):
    """特徴抽出から学習を回すためのクラス.
    """
    def __init__(self,
                 df_train,
                 df_test,
                 target_column,
                 splitter=SequenceSplitter(test_rate=0.1),
                 trainer=RandomForestTrainer(),
                 parameters=None,
                 drop_columns=[],
                 categorical_columns=[],
                 save_feature=False):
        self.base_train = df_train
        self.base_test = df_test
        self.train = df_train[target_column]  # 目的変数のカラムを一つだけ保持するため
        if target_column in df_test.columns:
            self.test = df_test[target_column]  # 目的変数のカラムを一つだけ保持するため
        else:
            self.test = pd.DataFrame()
        self.target_column = target_column
        self.objective_var = 'binary_class'
        self.trainer = trainer
        self.trainer.set_task_setting(self.target_column, self.objective_var)
        self.evaluator = None
        self.reporter = None
        self.pipe_list = []
        self.done_pipe_list = []
        self.parameters = parameters
        self.drop_columns = drop_columns
        self.categorical_columns = categorical_columns
        self.save_feature = save_feature

    def add_pipe(self, pipe):
        if type(pipe) != list:
            if pipe in self.pipe_list:
                return
            self.pipe_list.append(pipe)
            self.done_pipe_list.append(False)
            return
        for p in pipe:
            if p in self.pipe_list:
                continue
            self.pipe_list.append(p)
            self.done_pipe_list.append(False)

    def add_evaluator_and_reporter(self, evaluator, reporter):
        self.evaluator = evaluator
        self.reporter = reporter

    def _create_features(self, df_train, df_test):
        if self.save_feature:
            self._save_target(df_train, df_test)
        for index, feature_class in enumerate(self.pipe_list):
            if self.done_pipe_list[index]:
                continue
            # FIXME: もっとスマートにタスクの区別をする方法があれば修正
            feature_class.set_target_column(self.target_column)
            if feature_class.task_name == 'Preprocessor':
                df_train, df_test = feature_class.fit(
                    df_train, df_test).transform(save=self.save_feature)
            elif feature_class.task_name == 'Feature':
                train, test = feature_class.fit(
                    df_train, df_test).transform(save=self.save_feature)
                df_train = pd.concat([df_train, train], axis=1)
                df_test = pd.concat([df_test, test], axis=1)
            self.done_pipe_list[index] = True
        return df_train, df_test

    def _save_target(self, df_train, df_test):
        feather.write_dataframe(
            df_train[[self.target_column]],
            str(Path.cwd() / 'data' / 'target_var_train.ftr'))
        if self.target_column in df_test.columns:
            feather.write_dataframe(
                df_test[[self.target_column]],
                str(Path.cwd() / 'data' / 'target_var_test.ftr'))

    def run(self):
        df_train, df_test = self._create_features(self.base_train,
                                                  self.base_test)
        if self.drop_columns:
            df_train.drop(self.drop_columns, axis=1, inplace=True)
            df_test.drop(self.drop_columns, axis=1, inplace=True)

        df_train = df_train.drop(
            self.target_column,
            axis=1) if self.target_column in df_train.columns else df_train
        df_test = df_test.drop(
            self.target_column,
            axis=1) if self.target_column in df_test.columns else df_test
        self.train = pd.concat([self.train, df_train], axis=1)
        self.test = pd.concat([self.test, df_test], axis=1)

        self.trainer.fit(self.train,
                         categorical_columns=self.categorical_columns,
                         parameters=self.parameters)
        pred_target, pred_prob = self.trainer.predict(self.test)

        if self.evaluator and self.reporter:
            self.evaluator.set_objective_var(self.objective_var)
            score_dict = self.evaluator.evaluate(
                self.test[self.target_column].values, pred_target, pred_prob)
            self.reporter.report(self.pipe_list, score_dict, self.trainer)
        elif self.evaluator and not self.reporter:
            raise ValueError('Reporterがセットされていません。')
        return pred_target


class MultiClassPipeline(BinaryClassPipeline):
    def __init__(self,
                 df_train,
                 df_test,
                 target_column,
                 splitter=SequenceSplitter(test_rate=0.1),
                 trainer=RandomForestTrainer(),
                 drop_columns=[],
                 save_feature=False):
        super().__init__(target_column, trainer, drop_columns, save_feature)
        self.objective_var = 'regression'


class RegressionPipeline(BinaryClassPipeline):
    def __init__(self,
                 df_train,
                 df_test,
                 target_column,
                 splitter=SequenceSplitter(test_rate=0.1),
                 trainer=RandomForestTrainer(),
                 drop_columns=[],
                 save_feature=False):
        super().__init__(target_column, trainer, drop_columns, save_feature)
        self.objective_var = 'regression'
