from abc import ABCMeta, abstractmethod
import mlflow
import mlflow.sklearn


class InterfaceReporter(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def report(self, pipe_list, score_dict, model):
        pass


class Reporter(InterfaceReporter):
    def __init__(self):
        pass

    def report(self, pipe_list, score_dict, trainer):
        with mlflow.start_run() as run:
            for pipe in pipe_list:
                mlflow.log_param(pipe.name, pipe.columns)
            for key, score in score_dict.items():
                mlflow.log_metric(key, score)
            for key, param in trainer.parameters.items():
                mlflow.log_param(key, param)
            # TODO: mlflowでモデルを保存するための条件分岐を書く
            mlflow.sklearn.log_model(trainer.model, "model")