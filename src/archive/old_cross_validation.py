import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def cross_validate_model(X, y, model_class, model_params=None, n_splits=5, random_state=42):
    """
    performs manual K-Fold cross-validation for any sklearn regression model
    returns performance metrics and predictions
    """
    # create cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores = []
    rmse_scores = []
    mae_scores = []
    predictions = []

    # loops over each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = X.iloc[train_idx] # gets training and testing data by index
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # creates and trains model
        if model_params is None:
            model = model_class()
        else:
            model = model_class(**model_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # evaluates on test-fold
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        # computes metrics for each fold
        fold_df = pd.DataFrame({
            "y_test": y_test.values,
            "y_pred": y_pred,
            "fold": fold
        })
        predictions.append(fold_df)

        print(f"Fold {fold + 1} - R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    all_preds = pd.concat(predictions, ignore_index=True)

    return r2_scores, rmse_scores, mae_scores, all_preds