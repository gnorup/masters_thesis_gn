# run RandomForestRegressor or LinearRegression within a pre-defined stratified K-fold setup

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from config.constants import ID_COL

# run regression inside stratified 5-fold cross-validation
def stratified_cross_validation(
        df, fold_column, model_type, model_params, target_column, n_folds=5, feature_columns=None
):
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    all_preds = []

    for fold in range(1, n_folds + 1):
        train_df = df[df[fold_column] != fold] # all the other folds
        test_df = df[df[fold_column] == fold] # current fold

        # train-test-split
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        # standardize
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # train model
        model = model_type(**model_params) if model_params else model_type()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # model evaluation
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        fold_df = pd.DataFrame({
            ID_COL: test_df[ID_COL].values,
            "y_test": y_test.values,
            "y_pred": y_pred,
            "fold": fold,
            "model": model_type.__name__
        })
        all_preds.append(fold_df)

    all_preds_df = pd.concat(all_preds, ignore_index=True)

    return r2_scores, rmse_scores, mae_scores, all_preds_df
