
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def sample_random_parameters(param_grid):
    return {key: np.random.choice(values) for key, values in param_grid.items()}

# Function for Autoregressive Predictions
def autoregressive_predict(model, X_test, steps):
    predictions = []
    test_data = X_test.values  # Convert DataFrame to NumPy array for easier handling
    for step in range(steps):
        # Ensure we are using a single sample for prediction
        sample = test_data[step].reshape(1, -1)
        pred = model.predict(sample)[0]
        predictions.append(pred)
        # Update test_data for the next step
        if step + 1 < len(test_data):
            test_data[step + 1][:-1] = test_data[step][1:]
            test_data[step + 1][-1] = pred
    return predictions
# Function to Calculate Error Metrics
def calculate_error_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError

# Helper Functions
def load_dataset(data_id=42207, discard_nans=False):
    """
    Loads the 'Particulate Matter UKAIR 2017' dataset from OpenML by its dataset ID.

    This dataset contains hourly particulate matter air pollution data across Great Britain for 2017.
    The target variable is 'PM10 particulate matter', and the dataset includes other relevant features
    like date, time, and environmental conditions.

    Dataset Author: Ricardo Energy and Environment for DEFRA
    Source: OpenML and UK Department for Environment, Food and Rural Affairs (DEFRA)
    Accessed on: 2023-12-27

    Parameters:
    - data_id (int): The OpenML ID for the dataset. Default is 42207.
    - discard_nans (bool): Whether to discard NaN values from the dataset.

    Returns:
    - X (DataFrame): Features of the dataset.
    - y (Series): Target variable (PM10 particulate matter).
    """
    data_sk = fetch_openml(data_id=data_id, as_frame=True)
    X = data_sk.data
    y = data_sk.target

    # Convert 'datetime', 'Hour', and other time-related columns into a single datetime object
    X['datetime'] = pd.to_datetime(X['datetime']) + pd.to_timedelta(X['Hour'], unit='h')

    # Handling duplicate timestamps
    X.drop_duplicates(subset=['datetime'], inplace=True)
    y = y.reindex(X.index)

    if discard_nans:
        # Discard rows with NaN values in both X and y
        combined = X.join(y, how='inner')
        combined.dropna(inplace=True)
        X = combined[X.columns]
        y = combined[y.name]

    # Analyze the different values for 'Zone' and 'Environment.Type'
    unique_zones = X['Zone'].unique()
    unique_environments = X['Environment.Type'].unique()

    print(f"Unique Zones: {unique_zones}")
    print(f"Unique Environment Types: {unique_environments}")

    # Select a specific zone and environment type for analysis
    selected_zone = unique_zones[0]  # Replace with chosen zone
    selected_environment = unique_environments[0]  # Replace with chosen environment type

    # Filter the dataset based on the selected zone and environment
    filtered_X = X[(X['Zone'] == selected_zone) & (X['Environment.Type'] == selected_environment)]
    filtered_y = y.reindex(filtered_X.index)

    # Sort by datetime for chronological order
    filtered_X.sort_values('datetime', inplace=True)
    filtered_y = filtered_y.reindex(filtered_X.index)

    return filtered_X, filtered_y

def phase_space_embedding(X, embedding_dimension, time_delay, cut_off=False):
    """
    Transforms the dataset into a time series format using phase space embedding.
    This process involves reconstructing the time series data so that each data point
    is a vector containing 'embedding_dimension' number of values, spaced 'time_delay'
    time steps apart.

    Parameters:
    - X (DataFrame): The input time series dataset with a 'datetime' column.
    - embedding_dimension (int): The number of dimensions in the phase space.
    - time_delay (int): The time delay between points in the phase space.
    - cut_off (bool): Whether to cut off wrapped areas in the transformed dataset.

    Returns:
    - X_transformed (DataFrame): Transformed features dataset for time series prediction.
    - y_transformed (Series): Transformed target dataset for time series prediction.
    - X_datetime_transformed (DataFrame): Transformed datetime features dataset.
    - y_datetime_transformed (Series): Transformed datetime target dataset.
    """
    # Initialize empty DataFrames for the transformed data and datetime information
    X_transformed = pd.DataFrame()
    X_datetime_transformed = pd.DataFrame()

    # Adjusting for target variable
    for i in range(embedding_dimension):
        shifted_series = X['PM.sub.2.5..sub..particulate.matter..Hourly.measured.'].shift(-time_delay * i)
        shifted_datetime = X['datetime'].shift(-time_delay * i)

        if cut_off:
            shifted_series = shifted_series.iloc[:len(X) - time_delay * i]
            shifted_datetime = shifted_datetime.iloc[:len(X) - time_delay * i]

        X_transformed[f'dim_{i+1}'] = shifted_series
        X_datetime_transformed[f'dt_dim_{i+1}'] = shifted_datetime

    # Separating target variable
    y_transformed = X['PM.sub.2.5..sub..particulate.matter..Hourly.measured.'].shift(-time_delay * embedding_dimension)
    y_datetime_transformed = X['datetime'].shift(-time_delay * embedding_dimension)

    if cut_off:
        y_transformed = y_transformed.iloc[:len(X) - time_delay * embedding_dimension]
        y_datetime_transformed = y_datetime_transformed.iloc[:len(X) - time_delay * embedding_dimension]

    # Drop rows with missing values
    valid_indices = X_transformed.dropna().index.intersection(y_transformed.dropna().index)
    X_transformed = X_transformed.loc[valid_indices]
    X_datetime_transformed = X_datetime_transformed.loc[valid_indices]
    y_transformed = y_transformed.loc[valid_indices]
    y_datetime_transformed = y_datetime_transformed.loc[valid_indices]

    # Reset index for X_transformed and y_transformed
    X_transformed.reset_index(drop=True, inplace=True)
    y_transformed.reset_index(drop=True, inplace=True)
    X_datetime_transformed.reset_index(drop=True, inplace=True)
    y_datetime_transformed.reset_index(drop=True, inplace=True)

    return X_transformed, y_transformed, X_datetime_transformed, y_datetime_transformed

def hessian(x):
    """
    Calculates the second derivative of a given input.

    Parameters:
    x (numpy array): The input data.

    Returns:
    numpy array: The second derivative of the input data.
    """
    return np.gradient(np.gradient(x, axis=0), axis=0)

def calculate_variance_2nd_derivative(time_series, embedding_dimension=3, time_delay=1):
    """
    Calculate the variance of the second derivatives along a reconstructed phase space trajectory
    from a time series using Time Delay Embedding technique.

    Parameters:
    time_series (numpy array): The input time series data.
    embedding_dimension (int): The number of consecutive values in each row (default 3).
    time_delay (int): The number of time steps to shift each row (default 1).

    Returns:
    float: The variance of the second derivatives.
    """
    # Phase space embedding
    matrix = delay_embed(time_series, embedding_dimension, time_delay)

    # Second derivatives using Hessian
    second_derivatives = hessian(matrix)

    # Squaring, summing for each point, square root
    summed_squared = np.sqrt(np.sum(np.square(second_derivatives), axis=1))

    # Variance
    variance = np.var(summed_squared)

    return variance

def delay_embed(data, embedding_dimension=20, time_delay=1):
    """Delay embed data by concatenating consecutive increase delays.
    Parameters
    ----------
    data : array, 1-D
        Data to be delay-embedded.
    tau : int (default=10)
        Delay between subsequent dimensions (units of samples).
    max_dim : int (default=5)
        Maximum dimension up to which delay embedding is performed.
    Returns
    -------
    x : array, 2-D (samples x dim)
        Delay embedding reconstructed data in higher dimension.
    """
    if type(time_delay) is not int:
        time_delay = int(time_delay)

    num_samples = len(data) - time_delay * (embedding_dimension - 1)
    return np.array([data[dim * time_delay:num_samples + dim * time_delay] for dim in range(embedding_dimension)]).T[:,::-1]

def calculate_point_errors(y_true, y_pred):
    absolute_error = np.abs(y_true - y_pred)
    squared_error = np.square(y_true - y_pred)
    relative_error = np.abs((y_true - y_pred) / y_true)  # Handle division by zero if y_true contains zeros
    return absolute_error, squared_error, relative_error

def dfa_to_hurst(dfa):
    return dfa if dfa < 1 else dfa - 1

def categorize_hurst_intervals(hurst_values):
    intervals = pd.cut(hurst_values, bins=[0.0, 0.333, 0.666, 1.0], labels=['Low', 'Medium', 'High'])
    return intervals

def calculate_average_errors_per_interval(error_df):
    return error_df.groupby('hurst_interval').mean()

def calculate_errors_stats_per_interval(error_df):
    """
    Calculates the mean and standard deviation of errors and complexity measures for each Hurst interval.

    Parameters:
    - error_df (DataFrame): DataFrame containing error metrics and complexity measures.

    Returns:
    - DataFrame with mean and standard deviation for each error type and complexity measure, grouped by Hurst interval.
    """
    # Calculating mean and standard deviation
    stats_df = error_df.groupby('hurst_interval').agg(['mean', 'std'])

    # Renaming columns for clarity
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

    return stats_df

# Function to calculate average errors and standard deviations for each quartile of complexity measures
def calculate_errors_stats_per_quartile(error_df, complexity_quartiles, error_types):
    stats_per_quartile = {}
    for quartile in complexity_quartiles:
        stats_per_quartile[quartile] = error_df.groupby(quartile).agg(['mean', 'std'])[error_types].stack(level=0)
    return stats_per_quartile

def categorize_complexity_intervals_old(value, quantiles):
    if value <= quantiles[0.25]:
        return '1st Quartile'
    elif value <= quantiles[0.50]:
        return '2nd Quartile'
    elif value <= quantiles[0.75]:
        return '3rd Quartile'
    else:
        return '4th Quartile'

# Function to categorize data points into quartiles based on complexity metric
def categorize_complexity_intervals(value, quantiles):
    if value <= quantiles[0.25]:
        return 'Low'
    elif value <= quantiles[0.50]:
        return 'Mid-Low'
    elif value <= quantiles[0.75]:
        return 'Mid-High'
    else:
        return 'High'


def categorize_complexity_intervals(value, quantiles):
    if value <= quantiles[0.25]:
        return 'Low'
    elif value <= quantiles[0.75]:
        return 'Medium'
    else:
        return 'High'

# Function to calculate and print mean error statistics per quartile
def print_mean_error_stats_per_quartile(error_df, quartiles, error_types, quantiles_dict):
    for quartile in quartiles:
        print(f"Mean Error Statistics for {quartile} (Ranges: {quantiles_dict[quartile.replace('_quartile', '')]}):")
        for error_type in error_types:
            grouped_stats = error_df.groupby(quartile)[error_type].mean()
            print(f"{error_type}:")
            print(grouped_stats)
        print("\n")
