from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import numpy as np


def rmse_score(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def process_LinearRegression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_pred, y_test)
    lr_mse = mean_squared_error(y_pred, y_test)
    lr_evs = explained_variance_score(y_pred, y_test)
    lr_r2 = mean_absolute_error(y_pred, y_test)
    lr_rmse = rmse_score(y_pred, y_test)
    print("RMSE Score:", lr_rmse)
    lr_dict = {
        'lr_mae': round(lr_mae, 2),
        'lr_mse': round(lr_mse, 2),
        'lr_evs': round(lr_evs, 2),
        'lr_r2': round(lr_r2, 2),
        'lr_rmse': round(lr_rmse, 2)
    }
    return lr_dict


def process_decesionTree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dt_mae = mean_absolute_error(y_pred, y_test)
    dt_mse = mean_squared_error(y_pred, y_test)
    dt_evs = explained_variance_score(y_pred, y_test)
    dt_r2 = mean_absolute_error(y_pred, y_test)
    dt_rmse = rmse_score(y_pred, y_test)
    dt_dict = {
        'dt_mae': round(dt_mae, 2),
        'dt_mse': round(dt_mse, 2),
        'dt_evs': round(dt_evs, 2),
        'dt_r2': round(dt_r2, 2),
        'dt_rmse': round(dt_rmse, 2)
    }
    return dt_dict


def process_randomForest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_pred, y_test)
    rf_mse = mean_squared_error(y_pred, y_test)
    rf_evs = explained_variance_score(y_pred, y_test)
    rf_r2 = mean_absolute_error(y_pred, y_test)
    rf_rmse = rmse_score(y_pred, y_test)
    rf_dict = {
        'rf_mae': round(rf_mae, 2),
        'rf_mse': round(rf_mse, 2),
        'rf_evs': round(rf_evs, 2),
        'rf_r2': round(rf_r2, 2),
        'rf_rmse': round(rf_rmse, 2)
    }
    return rf_dict


def process_polynomialRegressor(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    lin = LinearRegression()
    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(X_train)
    x_t = poly.fit_transform(X_test)
    lin.fit(X_poly, y_train)
    y_pred = lin.predict(x_t)
    pf_mae = mean_absolute_error(y_pred, y_test)
    pf_mse = mean_squared_error(y_pred, y_test)
    pf_evs = explained_variance_score(y_pred, y_test)
    pf_r2 = mean_absolute_error(y_pred, y_test)
    pf_rmse = rmse_score(y_pred, y_test)
    pf_dict = {
        'pf_mae': round(pf_mae, 2),
        'pf_mse': round(pf_mse, 2),
        'pf_evs': round(pf_evs, 2),
        'pf_r2': round(pf_r2, 2),
        'pf_rmse': round(pf_rmse, 2)
    }
    return pf_dict
