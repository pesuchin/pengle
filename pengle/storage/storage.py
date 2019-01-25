from sklearn.externals import joblib


def save_model(model, output_path):
    joblib.dump(model, output_path)


def output_csv(df, output_path):
    df.to_csv(output_path, header=True, index=False)
