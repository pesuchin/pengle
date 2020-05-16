from pengle.transformer.base import Feature


class FundamentalStatistics(Feature):
    """groupbyからの基本統計量による特徴量生成用のクラス.

    例)
    >>> train, test = FundamentalStatistics().fit(
                        df_train, df_test, groupby_keys=['id'], agg_columns=['a', 'b']
                      ).transform()
    """
    def create_features(self, df_train, df_test, groupby_keys, agg_columns):
        """特徴量生成関数.

        Arguments:
            df_train {pd.DataFrame} -- 訓練データセット
            df_test {pd.DataFrame} -- テストデータセット
            columns {list} -- 特徴抽出を行う対象のクラス
        """
        agg_funcs = [
            "min", "max", "sum", "var", "std", "mean", "count", "median"
        ]
        grouping_funcs = {column: agg_funcs for column in agg_columns}
        train = df_train.groupby(groupby_keys).agg(agg_columns)
        test = df_test.groupby(groupby_keys).agg(agg_columns)
        for column in train.columns:
            self.train[column] = train[column]
            self.test[column] = test[column]
