from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np


class InterfaceEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def evaluate(self, true_target, pred_target):
        pass


def classification_eval(true_target, pred_target, pred_prob):
    score_dict = {}
    score_dict['acc'] = accuracy_score(true_target, pred_target)
    number_of_class = len(np.unique(true_target).tolist())
    if number_of_class == 2:
        score_dict['precision'] = precision_score(true_target, pred_target)
        score_dict['recall'] = recall_score(true_target, pred_target)
        score_dict['f1_score'] = f1_score(true_target, pred_target)
    elif number_of_class > 2:
        for avg_setting in [None, 'micro', 'macro', 'weighted']:
            score_dict['precision'] = precision_score(true_target,
                                                      pred_target,
                                                      average=avg_setting)
            score_dict['recall'] = recall_score(true_target,
                                                pred_target,
                                                average=avg_setting)
            score_dict['f1_score'] = f1_score(true_target,
                                              pred_target,
                                              average=avg_setting)
    else:
        raise ValueError('[Error] 目的変数の種類数が2より少ないため、データが誤っていないかご確認ください。')
    return score_dict


def regression_eval(true_target, pred_target, pred_prob):
    score_dict = {}
    score_dict['mae'] = mean_absolute_error(true_target, pred_target)
    score_dict['mse'] = mean_squared_error(true_target, pred_target)
    score_dict['msl'] = mean_squared_log_error(true_target, pred_target)
    return score_dict


class Evaluator(InterfaceEvaluator):
    def __init__(self):
        pass

    def set_objective_var(self, objective_var='binary_class'):
        self.objective_var = objective_var

    def evaluate(self, true_target, pred_target, pred_prob):
        if self.objective_var in ['binary_class', 'multi_class']:
            return classification_eval(true_target, pred_target, pred_prob)
        elif self.objective_var == 'regression':
            return regression_eval(true_target, pred_target, pred_prob)
        else:
            raise NotImplementedError
