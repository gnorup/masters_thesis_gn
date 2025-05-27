import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_and_evaluate_regression_model(
        X, y,
        model_class,
        model_params=None,
        test_size=0.2,
        random_state=42
):
    """
    trains and evaluates a regression model
    model_class options: LinearRegression, Ridge, Lasso, RandomForestRegressor
    model_params for Ridge and Lasso: {"alpha": x}
    model_params for Random Forest: {"n_estimators": x} (number of decision trees)
    returns: model, evaluation scores, data splits, predictions
    """

    # standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # create model
    if model_params is None:
        model = model_class()
    else:
        model = model_class(**model_params)

    # train model
    model.fit(X_train, y_train)

    # predict on train data (for plot later)
    y_pred_train = model.predict(X_train)

    # predict on test data
    y_pred_test = model.predict(X_test)

    # calculate evaluation metrics
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)

    metrics = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }

    return model, metrics, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
