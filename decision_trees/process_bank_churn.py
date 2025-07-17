import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, List, Optional


def drop_na_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop rows that contain missing values in specified columns.

    Args:
        df (pd.DataFrame): The raw DataFrame.
        columns (List[str]): List of column names to check for missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame without NA values in specified columns.
    """
    return df.dropna(subset=columns)


def create_inputs_targets(df_dict: dict, input_cols: List[str], target_col: str) -> dict:
    """
    Split input DataFrames into input features and target labels.

    Args:
        df_dict (dict): Dictionary containing train/validation DataFrames.
        input_cols (List[str]): List of input feature column names.
        target_col (str): Name of the target column.

    Returns:
        dict: Dictionary with separated inputs and targets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def impute_missing_values(data: dict, numeric_cols: List[str]) -> SimpleImputer:
    """
    Impute missing numeric values using mean strategy.

    Args:
        data (dict): Dictionary with train and validation inputs.
        numeric_cols (List[str]): List of numeric column names.

    Returns:
        SimpleImputer: Fitted SimpleImputer instance.
    """
    imputer = SimpleImputer(strategy='mean').fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(data[f'{split}_inputs'][numeric_cols])
    return imputer


def scale_numeric_features(data: dict, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (dict): Dictionary with train and validation inputs.
        numeric_cols (List[str]): List of numeric column names.

    Returns:
        MinMaxScaler: Fitted scaler.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])
    return scaler


def encode_categorical_features(data: dict, categorical_cols: List[str]) -> Tuple[OneHotEncoder, List[str]]:
    """
    Encode categorical columns using one-hot encoding.

    Args:
        data (dict): Dictionary with train and validation inputs.
        categorical_cols (List[str]): List of categorical column names.

    Returns:
        Tuple[OneHotEncoder, List[str]]: Fitted encoder and list of encoded column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'].drop(columns=categorical_cols), encoded_df], axis=1)

    return encoder, encoded_cols


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """
    Preprocess raw bank churn dataset.

    Steps:
    - Drop NA values in target column
    - Drop 'Surname' column
    - Split into train and validation sets
    - Impute missing numeric values
    - Optionally scale numeric features
    - One-hot encode categorical features

    Args:
        raw_df (pd.DataFrame): Raw input data.
        scaler_numeric (bool): Whether to scale numeric columns (useful for non-tree models).

    Returns:
        Tuple:
            - X_train (pd.DataFrame)
            - train_targets (pd.Series)
            - X_val (pd.DataFrame)
            - val_targets (pd.Series)
            - input_cols (List[str])
            - scaler (Optional[MinMaxScaler])
            - encoder (OneHotEncoder)
    """
    raw_df = drop_na_values(raw_df, ['Exited'])

    # Drop Surname to avoid overfitting on identity-related data
    if 'Surname' in raw_df.columns:
        raw_df = raw_df.drop(columns=['Surname'])

    # Train/val split
    train_df = raw_df.sample(frac=0.8, random_state=42)
    val_df = raw_df.drop(train_df.index)
    df_dict = {'train': train_df, 'val': val_df}

    target_col = 'Exited'
    input_cols = [col for col in raw_df.columns if col not in ['Exited']]

    data = create_inputs_targets(df_dict, input_cols, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    imputer = impute_missing_values(data, numeric_cols)
    scaler = None
    if scaler_numeric:
        scaler = scale_numeric_features(data, numeric_cols)

    encoder, encoded_cols = encode_categorical_features(data, categorical_cols)

    full_input_cols = numeric_cols + encoded_cols

    X_train = data['train_inputs'][full_input_cols]
    train_targets = data['train_targets']
    X_val = data['val_inputs'][full_input_cols]
    val_targets = data['val_targets']

    return X_train, train_targets, X_val, val_targets, full_input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool = True
) -> pd.DataFrame:
    """
    Preprocess new (test or prediction) data using already fitted scaler and encoder.

    Args:
        new_df (pd.DataFrame): New raw data to preprocess.
        input_cols (List[str]): List of expected input columns.
        scaler (Optional[MinMaxScaler]): Fitted scaler or None if scaling is skipped.
        encoder (OneHotEncoder): Fitted encoder.
        numeric_cols (List[str]): List of numeric columns.
        categorical_cols (List[str]): List of categorical columns.
        scale_numeric (bool): Whether to scale numeric features.

    Returns:
        pd.DataFrame: Preprocessed input DataFrame ready for prediction.
    """
    new_df = new_df.copy()

    if 'Surname' in new_df.columns:
        new_df = new_df.drop(columns=['Surname'])

    inputs = new_df[input_cols].copy()

    if scale_numeric and scaler is not None:
        inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])

    encoded = encoder.transform(inputs[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=inputs.index)
    inputs = pd.concat([inputs.drop(columns=categorical_cols), encoded_df], axis=1)

    return inputs
