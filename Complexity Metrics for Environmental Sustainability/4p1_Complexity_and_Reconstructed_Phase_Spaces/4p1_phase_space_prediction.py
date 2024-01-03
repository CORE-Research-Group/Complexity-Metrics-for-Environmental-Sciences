"""
This script is part of a research paper investigating the use of machine learning models, specifically CatBoost, to predict time series data related to air pollution. The dataset, 'Particulate Matter UKAIR 2017,' comprises hourly air pollution data across Great Britain.

Key steps in the script:
1. Import necessary libraries for data handling, machine learning, and phase space embedding.
2. Load and preprocess the 'Particulate Matter UKAIR 2017' dataset from OpenML.
3. Transform the dataset into a time series format suitable for phase space embedding.
4. Apply phase space embedding to restructure the dataset for time series prediction.
5. Split the transformed dataset into training and testing sets.
6. Initialize and tune a CatBoost Classifier using Bayesian Optimization.
7. Evaluate the model on the test set, focusing on time series prediction accuracy.
8. Save the model's performance metrics and feature importances for analysis.
"""

# Step 1: Import necessary libraries
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from skopt import BayesSearchCV
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize


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

    # Plot the time series data
    #plt.figure(figsize=(12, 6))
    #plt.plot(filtered_X['datetime'], filtered_y, label='PM10 Particulate Matter')
    #plt.title(f'Time Series Plot for {selected_zone}, {selected_environment}')
    #plt.xlabel('Datetime')
    #plt.ylabel('PM10 Particulate Matter')
    #plt.legend()
    #plt.show()

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

    return X_transformed, y_transformed, X_datetime_transformed, y_datetime_transformed


# Load and preprocess dataset
print('Loading Dataset')
X, y = load_dataset()
print('Dataset loaded')

# Extract the target time series
target_series = X['PM.sub.2.5..sub..particulate.matter..Hourly.measured.']

# Define parameters for phase space embedding
embedding_dimension = 3  # Define based on domain knowledge, e.g., false nearest neighbour algorihtm
time_delay = 1  # Define based on domain knowledge, e.g., based on minimal mutual information.
# However for many real life cases (data), embedding dimension = 3 and time delay = 1 works just fine as a first guess.
# Further, often the algorithms to determine the correct embedding are not working correctly for particular data sets. Thus we didn't include them here.
# Further improvement of the prediction can be achieved by increasing the number of iterations for the Bayesian Optimization of the Hyperparameters:
n_iter = 10
# n_iter > 50 might take a long time

# Transform the dataset
X_transformed, y_transformed, X_datetime_transformed, y_datetime_transformed = phase_space_embedding(X, embedding_dimension, time_delay)

# Extract transformed data for plotting
df_shifted = X_transformed.iloc[:1000]

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original Time Series Plot
sns.lineplot(x=X['datetime'], y=X['PM.sub.2.5..sub..particulate.matter..Hourly.measured.'], ax=axes[0])
axes[0].set_title(f'Original Time Series\nEmbedding Dimension: {embedding_dimension}, Time Delay: {time_delay}')
axes[0].set_xlabel('Datetime')
axes[0].set_ylabel('PM2.5 Level')

# Phase Space Embedded Plot
if embedding_dimension >= 3:
    # 3D plot with color gradient
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    points = np.array([df_shifted['dim_1'], df_shifted['dim_2'], df_shifted['dim_3']]).T.reshape(-1, 1, 3)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous variable (e.g., range of indices) for color mapping
    num_segs = segs.shape[0]
    color_values = np.linspace(0, 1, num_segs)

    # Create Line3DCollection with color mapping
    lc = Line3DCollection(segs, cmap='viridis')
    lc.set_array(color_values)  # Set the color mapping

    ax.add_collection3d(lc)
    ax.set_xlim(df_shifted['dim_1'].min(), df_shifted['dim_1'].max())
    ax.set_ylim(df_shifted['dim_2'].min(), df_shifted['dim_2'].max())
    ax.set_zlim(df_shifted['dim_3'].min(), df_shifted['dim_3'].max())
    ax.set_title(f'Phase Space Embedded Time Series (3D Projection with Color Gradient)\nEmbedding Dimension: {embedding_dimension}, Time Delay: {time_delay}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
else:
    # 2D plot for 2 dimensions
    sns.scatterplot(x=df_shifted['dim_1'], y=df_shifted['dim_2'], ax=axes[1])
    axes[1].set_title(f'Phase Space Embedded Time Series (2D Projection)\nEmbedding Dimension: {embedding_dimension}, Time Delay: {time_delay}')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')

# Save and show the plot
plot_filename = f'time_series_phase_space_embedding_{embedding_dimension}_td{time_delay}.png'
plt.tight_layout()
plt.savefig(plot_filename)
plt.show()

# After phase space embedding
feature_names = [f'dim_{i+1}' for i in range(embedding_dimension)]

# Step 5: Split the data
print('Performing Train Test Split')
# Split the feature data
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y_transformed, test_size=0.2, shuffle=False, random_state=42)

# Split the datetime information in the same manner
datetime_train, datetime_test = train_test_split(
    X_datetime_transformed, test_size=0.2, shuffle=False, random_state=42)

# Step 5: Define the parameter range for Bayesian Optimization
parameter_grid_cb = {
    'depth': [4, 5, 6, 7, 8, 9, 10, 11, 12],
    'learning_rate': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.75, 0.8125,
                      0.9],
    'iterations': [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [16, 20, 32, 40, 50, 64, 70, 80, 90, 100, 110, 120, 128, 200, 256],
    'bagging_temperature': [0, 1, 2, 3, 4]
}

# Step 4: Initialize CatBoost Classifier
model = CatBoostRegressor(verbose=False)

# Initialize Bayesian Optimization with CatBoost
bocv = BayesSearchCV(model, parameter_grid_cb, n_iter=n_iter, cv=5, verbose=1)

# Step 6: Fit the model to the training data
print('Training CatBoost Regressor')
bocv.fit(X_train, y_train)

# Use the best estimator
model = bocv.best_estimator_
best_params = bocv.best_params_
best_cv_score = bocv.best_score_

print("Best Parameters:", best_params)
print("Best CV Score:", best_cv_score)

# Step 7: Model evaluation
predictions = model.predict(X_test)

# Calculate performance metrics for regression

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Scores:")
print("========================================")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")
print("========================================\n")

# Step 8: Feature importance analysis
feature_importances = model.get_feature_importance()
sorted_feature_importances = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

print("Sorted Feature Importances:")
for name, importance in sorted_feature_importances:
    print(f"{name}: {importance}")

feature_importances_dict = {name: importance for name, importance in sorted_feature_importances}

# Step 9: Prepare and Save Results
embedding_label = f"td{time_delay}_ed{embedding_dimension}"  # Time delay and embedding dimension label
results = {
    "best_parameters": best_params,
    "best_cv_score": best_cv_score,
    "mean_squared_error": mse,
    "mean_absolute_error": mae,
    "r2_score": r2,
    "feature_importances": feature_importances_dict,
    "phase_space_embedding": {
        "embedding_dimension": embedding_dimension,
        "time_delay": time_delay
    }
}

filename = f"particulate_matter_regression_{embedding_label}_results.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {filename}")

# Step 10: Visualize the Predictions
print('Visualizing Predictions')

# Flatten datetime information for merging
datetime_test_flattened = datetime_test.apply(lambda row: row.dropna().iloc[0], axis=1)

# Reconstruct the original time series
original_ts = pd.DataFrame({
    'Datetime': datetime_test_flattened,
    'Original': y_test
})

# Reconstruct the predicted time series
predicted_ts = pd.DataFrame({
    'Datetime': datetime_test_flattened,
    'Predicted': predictions
})

# Merge the original and predicted data for plotting
plot_data = pd.merge(original_ts, predicted_ts, on='Datetime')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(plot_data['Datetime'], plot_data['Original'], label='Original', alpha=0.7)
plt.plot(plot_data['Datetime'], plot_data['Predicted'], label='Predicted', alpha=0.7)
plt.title('Original vs Predicted PM10 Particulate Matter Levels')
plt.xlabel('Datetime')
plt.ylabel('PM10 Level')
plt.legend()

# Save plot
plot_filename = f'time_series_prediction_plot_td{time_delay}_ed{embedding_dimension}.png'
plt.savefig(plot_filename)
print(f'Plot saved as {plot_filename}')

# Now show the plot
plt.show()

# Update results JSON file
results['time_series_plot'] = plot_filename
with open(filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Updated results saved to {filename}")


