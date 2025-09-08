import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.models import NaiveSeasonal, NaiveDrift, XGBModel, ExponentialSmoothing, ARIMA, AutoARIMA, RNNModel 
from darts.metrics import mae as mae_metric, mape as mape_metric, rmse
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from functools import reduce
import itertools 
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error 
warnings.filterwarnings('ignore')



def plot_forecasts(train: TimeSeries, val: TimeSeries = None, preds: dict = None, title: str = "Forecast Plot"):
    """
    Unified function to plot train, validation, and forecast data using Plotly.

    Parameters:
    - train (TimeSeries): The Darts TimeSeries object representing the training data.
    - val (TimeSeries, optional): The Darts TimeSeries object for validation data.
    - preds (dict, optional): A dictionary of Darts TimeSeries predictions, where keys are model names.
    - title (str): The title of the plot.
    """
    fig = go.Figure()

    # Plot the training data
    fig.add_trace(go.Scatter(x=train.time_index, y=train.values().flatten(),
                             mode='lines', name='Train', line=dict(width=2)))

    # Plot the validation data if provided
    if val is not None:
        fig.add_trace(go.Scatter(x=val.time_index, y=val.values().flatten(),
                                 mode='lines', name='Validation', line=dict(width=2)))

    # Plot predictions from a dictionary of forecasts
    if preds is not None:
        if isinstance(preds, dict):
            for name, pred in preds.items():
                fig.add_trace(go.Scatter(x=pred.time_index, y=pred.values().flatten(),
                                         mode='lines', name=name))
        else:
            # Handle a single Darts TimeSeries object prediction
            fig.add_trace(go.Scatter(x=preds.time_index, y=preds.values().flatten(),
                                     mode='lines', name='Forecast', line=dict(dash='dash')))

    # Update layout for a clean, informative plot
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white'
    )
    fig.show()


def fit_eval_model(model, model_name: str, train: TimeSeries, val: TimeSeries):
    """
    Fits a Darts model and evaluates its performance on the validation set.

    Parameters:
    - model: A Darts model instance to be fitted and evaluated.
    - model_name (str): The name of the model for display purposes.
    - train (TimeSeries): The training time series data.
    - val (TimeSeries): The validation time series data.

    Returns:
    - tuple: A tuple containing the MAE, MAPE, and the forecast time series.
    """
    model.fit(train)
    forecast = model.predict(len(val))
    mae_val = mae_metric(val, forecast)
    mape_val = mape_metric(val, forecast)
    print(f"{model_name} -> MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")
    return mae_val, mape_val, forecast


def append_results(results_df: pd.DataFrame, model_name: str, mae_val: float, mape_val: float):
    """
    Appends the evaluation metrics of a model to a results DataFrame.

    Parameters:
    - results_df (pd.DataFrame): The DataFrame to which results will be appended.
    - model_name (str): The name of the model.
    - mae_val (float): The Mean Absolute Error (MAE) value.
    - mape_val (float): The Mean Absolute Percentage Error (MAPE) value.

    Returns:
    - pd.DataFrame: The updated results DataFrame.
    """
    results_df.loc[model_name] = [mae_val, mape_val]
    return results_df


def mae(actual: TimeSeries, predicted: TimeSeries):
    """
    Calculates the Mean Absolute Error (MAE) between two TimeSeries.

    Parameters:
    - actual (TimeSeries): The true values.
    - predicted (TimeSeries): The predicted values.

    Returns:
    - float: The calculated MAE.
    """
    actual_vals = actual.values().flatten()
    predicted_vals = predicted.values().flatten()
    return np.mean(np.abs(actual_vals - predicted_vals))


def mape(actual: TimeSeries, predicted: TimeSeries):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between two TimeSeries.
    Handles division by zero by masking out zero actual values.

    Parameters:
    - actual (TimeSeries): The true values.
    - predicted (TimeSeries): The predicted values.

    Returns:
    - float: The calculated MAPE as a percentage.
    """
    actual_vals = actual.values().flatten()
    predicted_vals = predicted.values().flatten()
    
    # Avoid division by zero by creating a mask
    mask = actual_vals != 0
    if np.sum(mask) == 0:
        return 0.0
    
    return np.mean(np.abs((actual_vals[mask] - predicted_vals[mask]) / actual_vals[mask])) * 100


def evaluate_xgb_params(train_data: TimeSeries, val_data: TimeSeries, params: dict, covariates_train: TimeSeries, covariates_full_for_pred: TimeSeries):
    """
    Evaluates an XGBModel with the given parameters, including covariates.

    Parameters:
    - train_data (TimeSeries): The training time series data.
    - val_data (TimeSeries): The validation time series data.
    - params (dict): A dictionary of parameters for the XGBModel.
    - covariates_train (TimeSeries): Covariates aligned with the training data.
    - covariates_full_for_pred (TimeSeries): Full covariates for making predictions.

    Returns:
    - tuple: A tuple containing the MAE score (float) and the fitted XGBModel.
    """
    model = XGBModel(
        lags=params.get('lags', list(range(-21, 0))),
        lags_past_covariates=params.get('lags_past_covariates', list(range(-21, 0))),
        output_chunk_length=1,
        n_estimators=params.get('n_estimators', 200),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        random_state=42
    )
    
    # Train the model using the provided covariates_train
    model.fit(series=train_data, past_covariates=covariates_train)
    
    # Make a prediction using the full covariates
    forecast = model.predict(n=len(val_data), past_covariates=covariates_full_for_pred)
    
    # Calculate the MAE metric
    mae_val = mae_metric(val_data, forecast)

    return mae_val, model

def simple_grid_search(
    train_data: TimeSeries,
    val_data: TimeSeries,
    covariates_train: TimeSeries,
    covariates_full_for_pred: TimeSeries,
    max_trials: int = 30
):
    """
    Performs a simple grid search for XGBModel with covariates.
    ...
    """
    print("Starting Simple Grid Search...")
    
    param_options = {
        'lags': [7, 14, 21, 30],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.1, 0.15, 0.2]
    }
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    param_names = list(param_options.keys())
    param_values = list(param_options.values())
    all_combinations = list(itertools.product(*param_values))
    test_combinations = all_combinations[:max_trials]
    
    print(f"Testing {len(test_combinations)} parameter combinations...")
    
    # Pass the covariates to the evaluation function here
    for i, combination in enumerate(test_combinations):
        params = dict(zip(param_names, combination))
        score, model = evaluate_xgb_params(
            train_data,
            val_data,
            params,
            covariates_train=covariates_train,
            covariates_full_for_pred=covariates_full_for_pred
        )
        
        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_model = model
            print(f"New best score: {best_score:.4f} with params: {best_params}")
    
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{len(test_combinations)} combinations")
    
    print("Grid Search Complete!")
    print(f"Best MAE: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, best_score, best_model