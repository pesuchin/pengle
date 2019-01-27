import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from pengle.storage.storage import save_model, output_csv


def get_default_parametor(objective):
    if objective == 'regression':
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2'},
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
    elif objective == 'classification':
        params = {'learning_rate': 0.03,
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

    return params


def train_cv(x, y, lgb_params,
             number_of_folds=5, 
             evaluation_metric='auc', 
             save_feature_importances=True, 
             early_stopping_rounds=50, 
             num_round=50,
             random_state=19930201,
             shuffle=True,
             drop_columns=[],
             categorical_columns=[],
             score_metric='auc'):
    """Cross Validation用の関数

    Arguments:
        x {DataFrame} -- 学習用の特徴量のDataFrame
        y {list} -- 学習用の目的変数のリスト
        lgb_params {dict} -- lightgbmのパラメータの辞書

    Keyword Arguments:
        number_of_folds {int} -- StratifiedKFoldのsplitの数 (default: {5})
        evaluation_metric {str} -- [description] (default: {'auc'})
        save_feature_importances {bool} -- [description] (default: {True})
        early_stopping_rounds {int} -- [description] (default: {50})
        num_round {int} -- [description] (default: {50})
        random_state {int} -- [description] (default: {19930201})
        shuffle {bool} -- [description] (default: {True})
        drop_columns {list} -- [description] (default: {[]})
        categorical_columns {list} -- [description] (default: {[]})
        score_metric {str} -- [description] (default: {'auc'})

    Returns:
        [type] -- [description]
    """

    cross_validator = StratifiedKFold(n_splits=number_of_folds,
                                      random_state=random_state,
                                      shuffle=shuffle)

    validation_scores = []
    models = []
    feature_importance_df = pd.DataFrame()
    y = np.array(y)
    cv = cross_validator.split(x, y)
    for fold_index, (train_index, validation_index) in enumerate(cv):
        x_train, x_validation = x.iloc[train_index], x.iloc[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        if drop_columns:
            x_train.drop(drop_columns, axis=1, inplace=True)
            x_validation.drop(drop_columns, axis=1, inplace=True)

        x_train_columns = x_train.columns
        trn_data = lgb.Dataset(x_train,
                               label=y_train,
                               categorical_feature=categorical_columns)
        del x_train
        del y_train
        val_data = lgb.Dataset(x_validation,
                               label=y_validation,
                               categorical_feature=categorical_columns)
        model = lgb.train(lgb_params,
                          trn_data,
                          num_round,
                          valid_sets=[trn_data, val_data],
                          verbose_eval=100,
                          early_stopping_rounds=early_stopping_rounds
                          )

        models.append(model)

        predictions = model.predict(x_validation, num_iteration=model.best_iteration)

        score = calc_score(score_metric, y_validation, predictions)

        validation_scores.append(score)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = x_train_columns
        fold_importance_df["importance"] = model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = fold_index + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    if save_feature_importances:
        save_importance_graph(feature_importance_df)

    score = sum(validation_scores) / len(validation_scores)
    return models, score


def calc_score(score_metric, y, predictions):
    if score_metric == 'auc':
        false_positive_rate, recall, thresholds = metrics.roc_curve(y, predictions)
        score = metrics.auc(false_positive_rate, recall)
    elif score_metric == 'accuracy':
        predictions = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
        score = metrics.accuracy_score(y, predictions)
    elif score_metric == 'f1':
        predictions = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
        score = metrics.f1_score(y, predictions)
    elif score_metric == 'precision':
        predictions = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
        score = metrics.precision_score(y, predictions)
    elif score_metric == 'recall':
        predictions = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]
        score = metrics.recall_score(y, predictions)
    elif score_metric == 'mae':
        score = metrics.mean_absolute_error(y, predictions)
    elif score_metric == 'mse':
        score = metrics.mean_squared_error(y, predictions)
    elif score_metric == 'msle':
        score = metrics.mean_squared_log_error(y, predictions)
    return score


def save_importance_graph(feature_importance_df):
    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:1000].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                                ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

    # mean_gain = feature_importances[['gain', 'feature']].groupby('feature').mean()
    # feature_importances['mean_gain'] = feature_importances['feature'].map(mean_gain['gain'])
    #
    # temp = feature_importances.sort_values('mean_gain', ascending=False)
    best_features.sort_values(by="importance", ascending=False) \
        .groupby("feature") \
        .mean() \
        .sort_values(by="importance", ascending=False) \
        .to_csv('feature_importances_new.csv', index=True)
    return best_features


def train(x, y, lgb_params,
          train_size=100,
          early_stopping_rounds=50, 
          num_round=50,
          random_state=19930201,
          drop_columns=[],
          categorical_columns=[]):
    x_train, x_validation = x.iloc[:train_size, :], x.iloc[train_size:, :]
    y_train, y_validation = y[:train_size], y[train_size:]

    if drop_columns:
        x_train.drop(drop_columns, axis=1, inplace=True)
        x_validation.drop(drop_columns, axis=1, inplace=True)

    x_train_columns = x_train.columns
    trn_data = lgb.Dataset(x_train,
                           label=y_train,
                           categorical_feature=categorical_columns)
    del x_train
    del y_train
    val_data = lgb.Dataset(x_validation,
                           label=y_validation,
                           categorical_feature=categorical_columns)
    model = lgb.train(lgb_params,
                      trn_data,
                      num_round,
                      valid_sets=[trn_data, val_data],
                      verbose_eval=100,
                      early_stopping_rounds=early_stopping_rounds
                      )

    predictions = model.predict(x_validation, num_iteration=model.best_iteration)
    save_model(model, './output/models/')


def train_and_predict_test(X, y, X_test, lgb_params, model_name, id_col, predict_col_name,
                           predict_method='binary',
                           train_size=600,
                           early_stopping_rounds=50, 
                           num_round=50,
                           random_state=19930201,
                           drop_columns=[],
                           categorical_columns=[]):
    X_train, X_validation = X.iloc[:train_size, :], X.iloc[train_size:, :]
    y_train, y_validation = y[:train_size], y[train_size:]

    if drop_columns:
        X_train.drop(drop_columns, axis=1, inplace=True)
        X_validation.drop(drop_columns, axis=1, inplace=True)
        X_test.drop(drop_columns, axis=1, inplace=True)

    x_train_columns = X_train.columns

    train_data = lgb.Dataset(X_train,
                             label=y_train,
                             categorical_feature=categorical_columns)
    del X_train
    del y_train
    val_data = lgb.Dataset(X_validation,
                           label=y_validation,
                           categorical_feature=categorical_columns)

    model = lgb.train(lgb_params,
                      train_data,
                      num_round,
                      valid_sets=[train_data, val_data],
                      verbose_eval=100,
                      early_stopping_rounds=early_stopping_rounds
                      )

    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    if predict_method == 'binary':
        predictions = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]

    df_submit = pd.DataFrame()
    df_submit[id_col.name] = id_col
    df_submit[predict_col_name] = predictions
    save_model(model, './output/models/' + model_name + '_submit.ftr')
    output_csv(df_submit, './output/submits/' + model_name + '_submit.csv')
