"""
This script is developed as part of the research study 'Potentials and limitations of complexity research in Farming 4.0,'
to be featured in Current Opinion in Environmental Sustainability, 2024 by Mallinger et al. It is aligned with Section 4
of the research, focusing on the integration of complexity metrics and reconstructed phase spaces in the context of machine
learning for air pollution prediction. This approach is inspired by the methodologies and findings from two key papers:

1. "Reconstructed Phase Spaces and LSTM Neural Network Ensemble Predictions" by Sebastian Raubitzek and Thomas Neubauer
   (Eng. Proc. 2022, 18(1), 40; https://doi.org/10.3390/engproc2022018040). This paper explores combining reconstructed
   phase spaces with neural network predictions to enhance the predictability of time series data.

2. "Taming the Chaos in Neural Network Time Series Predictions" by Sebastian Raubitzek and Thomas Neubauer
   (Entropy 2021, 23(11), 1424; https://doi.org/10.3390/e23111424). This research presents a novel approach combining
   interpolation techniques, LSTM neural networks, and complexity measures for time series prediction.

The script is structured into the following key steps:

1. Import Libraries: Essential Python libraries for data manipulation, machine learning, and phase space embedding are integrated.
2. Set Parameters: Critical parameters for phase space embedding, such as embedding dimensions and time delay, are established.
3. Data Loading and Preprocessing: 'Particulate Matter UKAIR 2017' dataset is prepared, focusing on hourly air pollution data across Great Britain.
4. Phase Space Embedding: The dataset is transformed for time series analysis through phase space embedding.
5. Data Splitting: The transformed dataset is divided into training and testing subsets, essential for machine learning modeling.
6. Model Initialization and Tuning: A CatBoost Regressor is set up and fine-tuned using Bayesian Optimization for optimal prediction accuracy.
7. Model Evaluation: The model's performance on the test set is thoroughly assessed using metrics like mean squared error and R² score.
8. Visualization of Model Predictions: Predictive performance is graphically represented, comparing predicted results with actual data.

This script reflects a meticulous focus on applying machine learning techniques within environmental sustainability, underscoring the overarching theme of the research study.
"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 1: Import necessary libraries

# Machine learning algorithm for regression.
from catboost import CatBoostRegressor

# Function to split datasets into training and testing sets.
from sklearn.model_selection import train_test_split

# NumPy for numerical operations and array manipulations.
import numpy as np

# Pandas for data manipulation and handling.
import pandas as pd

# Nolds for calculating the fractal dimension of time series data.
import nolds

# JSON for parsing and outputting data in JSON format.
import json

# Matplotlib for creating static, interactive, and animated visualizations in Python.
import matplotlib.pyplot as plt

# Custom functions for prediction filtering and related operations.
import functions_prediction_filtering

import matplotlib.lines as mlines

########################################################################################################################
# SETTTING PARAMETERS ##################################################################################################
########################################################################################################################
# step 2: Setting parameters for machine learning model and data processing.

# Bayesian Optimization parameters for tuning machine learning models:
# 'n_iter' represents the number of iterations for Bayesian Optimization.
# More iterations can lead to better-tuned hyperparameters but also increase computational time.
# It is a key parameter in determining the extent of the search for optimal model parameters.
n_iter = 25  # Number of different predictions to be generated.

# Number of predictions to be considered for creating an ensemble model.
# An ensemble model combines multiple individual predictions to potentially improve the overall prediction accuracy.
n_ensemble = 3

# Percentage of data to be used for prediction, indicating the ratio of the train-test split.
# For example, a value of 5 indicates that 5% of the data will be used for testing the model,
# and the remaining 95% will be used for training the model.
# This parameter is crucial in determining how much data the model is trained on versus how much it is tested on.
prediction_per = 1

# Parameters that should remain constant for effective model functionin:

# 'embedding_dimension' is a key parameter in phase space reconstruction.
# It defines the number of past observations to consider for predicting future values.
# The choice of embedding dimension is crucial as it determines the structure of the phase space,
# which in turn influences the model's ability to capture the dynamics of the time series.
# While this parameter can be set based on domain knowledge or empirical methods such as the false nearest neighbors algorithm,
# Changing this value may affect the model's effectiveness in capturing the complexity of the data.
embedding_dimension = 3

# 'time_delay' refers to the delay between observations in the reconstructed phase space.
# This parameter is significant in phase space embedding as it impacts how the temporal structure of the data is represented.
# Time delay can be determined based on methods like minimal mutual information or autocorrelation analysis.
# For this model, the time delay is set to 1 and should be kept constant.
# Altering this parameter could lead to a different representation of the time series in phase space,
# potentially affecting the model's predictive performance.
time_delay = 1

########################################################################################################################
########################################################################################################################
########################################################################################################################

# Step 3: Load and Preprocess Dataset for Phase Space Embedding

# Loading the dataset
print('Loading Dataset')
X, y = functions_prediction_filtering.load_dataset()  # Load the 'Particulate Matter UKAIR 2017' dataset
print('Dataset loaded')

# Extract the target time series for phase space embedding
# Here, we're focusing on the 'PM2.5 particulate matter' as our target variable
target_series = X['PM.sub.2.5..sub..particulate.matter..Hourly.measured.'].values
target_series_datetime = X['datetime'].values

print(target_series)
#original_fractal_dimension = nolds.dfa(target_series[:int((len(target_series)*(1.0 - ((prediction_per * 1.25)/100))))])
original_fractal_dimension = functions_prediction_filtering.calculate_variance_2nd_derivative(target_series[:int((len(target_series)*(1.0 - ((prediction_per * 1.25)/100))))])

print(f'fractal dimension orignal dataset {original_fractal_dimension}')

# Transform the dataset using phase space embedding
# This step restructures the data based on the calculated time delay and embedding dimension
X_transformed, y_transformed, X_datetime_transformed, y_datetime_transformed = functions_prediction_filtering.phase_space_embedding(
    X, embedding_dimension, time_delay)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 5: Preparing the machine learning environment for model training and prediction.

# Performing the Train-Test Split
print('Performing Train Test Split')
# The transformed feature data is divided into training and testing sets.
# A specific portion of the data, determined by the 'prediction_per' parameter, is allocated for testing.
# This split is critical for evaluating the model's performance on unseen data while ensuring the temporal order is maintained.
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y_transformed, test_size=(prediction_per/100), shuffle=False, random_state=42)

# Splitting datetime information similarly to ensure chronological consistency.
# This step aligns the datetime data with the corresponding feature and target data, crucial for time series analysis.
datetime_train, datetime_test, y_datetime_train, y_datetime_test = train_test_split(
    X_datetime_transformed, y_datetime_transformed, test_size=(prediction_per/100), shuffle=False, random_state=42)

# This grid provides a range of hyperparameters for the model to explore during prediction.
parameter_grid_cb = {
    'depth': [4, 5, 6, 7, 8, 9, 10, 11, 12],
    'learning_rate': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.75, 0.8125, 0.9],
    'iterations': [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [16, 20, 32, 40, 50, 64, 70, 80, 90, 100, 110, 120, 128, 200, 256],
    'bagging_temperature': [0, 1, 2, 3, 4]
}

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 6: Autoregressive Machine Learning Predictions

# Initializing DataFrames for storing predictions, model parameters, and fractal dimensions.
prediction_df = pd.DataFrame()
params_df = pd.DataFrame()
fractal_dimensions_df = pd.DataFrame()

# Main loop for parameter sampling, autoregressive prediction, and fractal dimension calculation.
for iteration in range(n_iter):
    print(f'Prediction Iteration: {iteration}')
    sampled_params = functions_prediction_filtering.sample_random_parameters(parameter_grid_cb)
    model = CatBoostRegressor(**sampled_params, verbose=False)
    model.fit(X_train, y_train)
    preds = functions_prediction_filtering.autoregressive_predict(model, X_test, len(X_test))
    prediction_df[f'iteration_{iteration}'] = preds
    params_df[f'iteration_{iteration}'] = pd.Series(sampled_params)
    fractal_dim = functions_prediction_filtering.calculate_variance_2nd_derivative(preds)
    #fractal_dim = nolds.dfa(preds)
    fractal_dimensions_df[f'iteration_{iteration}'] = [fractal_dim]

# Computing delta fractal dimensions - the absolute difference between the original and predicted fractal dimensions.
fractal_delta_df = fractal_dimensions_df.subtract(original_fractal_dimension).abs()

# Selecting predictions for ensemble creation based on the smallest delta fractal dimensions.
selected_iterations = fractal_delta_df.T.nsmallest(n_ensemble, columns=0).index.tolist()

# Creating the ensemble prediction by averaging the selected predictions.
ensemble_predictions = prediction_df[selected_iterations].apply(lambda x: x.mean(), axis=1)

# Calculating the average of all predictions across different iterations.
average_all_predictions = prediction_df.mean(axis=1)

# DataFrames construction for different prediction sets:

# 1. Ensemble Prediction DataFrame.
ensemble_prediction_df = pd.DataFrame({
    'ensemble_prediction': ensemble_predictions
})

# 2. Average of All Predictions DataFrame.
average_all_predictions_df = pd.DataFrame({
    'average_prediction': average_all_predictions
})

# 3. Selected Predictions DataFrame.
selected_predictions_df = prediction_df[selected_iterations]

# 4. Average of Selected Predictions DataFrame.
average_selected_predictions_df = pd.DataFrame({
    'average_selected_prediction': ensemble_predictions
})

# Adding datetime to each DataFrame.
ensemble_prediction_df['datetime'] = y_datetime_test.values
average_all_predictions_df['datetime'] = y_datetime_test.values
selected_predictions_df['datetime'] = y_datetime_test.values
average_selected_predictions_df['datetime'] = y_datetime_test.values
prediction_df['datetime'] = y_datetime_test.values

# Printing DataFrames for verification.


print('#################################')
print(ensemble_prediction_df)
print('#################################')
print(average_all_predictions_df)
print('#################################')
print(selected_predictions_df)
print('#################################')
print(average_selected_predictions_df)
print('#################################')
print(prediction_df)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 7: Validation

# Implementing Validation for the Machine Learning Model.

# Calculating Error Metrics for Ensemble Predictions.
# The Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score are computed for the ensemble predictions.
ensemble_mse, ensemble_mae, ensemble_r2 = functions_prediction_filtering.calculate_error_metrics(y_test, ensemble_predictions)

# Calculating Error Metrics for the Average of All Predictions.
# The MSE, MAE, and R² Score are also computed for the average of all predictions.
average_mse, average_mae, average_r2 = functions_prediction_filtering.calculate_error_metrics(y_test, average_all_predictions)

# Displaying the Evaluation Results.
print("\nEvaluation of Prediction Methods:")
print("========================================")
print("Ensemble Predictions:")
print(f"Mean Squared Error: {ensemble_mse:.4f}")
print(f"Mean Absolute Error: {ensemble_mae:.4f}")
print(f"R² Score: {ensemble_r2:.4f}")
print("\nAverage of All Predictions:")
print(f"Mean Squared Error: {average_mse:.4f}")
print(f"Mean Absolute Error: {average_mae:.4f}")
print(f"R² Score: {average_r2:.4f}")
print("========================================\n")

# Saving the Evaluation Results.
# The results of the evaluation, including the error metrics for both ensemble and average predictions, are saved to a JSON file.
evaluation_results = {
    "ensemble_predictions": {
        "mse": ensemble_mse,
        "mae": ensemble_mae,
        "r2": ensemble_r2
    },
    "average_all_predictions": {
        "mse": average_mse,
        "mae": average_mae,
        "r2": average_r2
    }
}

with open('evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=4)

print("Evaluation results saved to 'evaluation_results.json'")
########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 8: Visualization

# Define the interval for short interval plots (20% of train data)
interval_length = int(len(y_test) * 1.2)
interval_start = len(target_series) - interval_length
short_interval_original = target_series[interval_start:]
short_interval_datetime = target_series_datetime[interval_start:]
last_timestamp_train = y_datetime_train.iloc[-1]

# Determine the index where train and test data split
split_index = len(target_series) - len(y_test)

# Prepare colors for each prediction in the upper right plot
colors = plt.cm.viridis(np.linspace(0, 1, n_iter))
print(np.shape(colors))
print(np.shape(prediction_df))
plt.figure(figsize=(15, 12))

# Visualizing the Results of our prediction analysis
plt.figure(figsize=(15, 12))

# Upper Left Plot: Full Length of Original Time Series.
plt.subplot(2, 2, 1)
plt.plot(target_series_datetime, target_series, color='black', label='Original Time Series')
plt.axvline(x=last_timestamp_train, color='purple', linestyle='--', label='Train-Test Split')
plt.title('Full Length of Original Time Series')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()

# Upper Right Plot: Test Interval with All Predictions.
plt.subplot(2, 2, 2)
plt.plot(short_interval_datetime, short_interval_original, color='black', label='Original')
plt.axvline(x=last_timestamp_train, color='purple', linestyle='--')
# Plotting the predictions
for i, col in enumerate(colors):
    plt.plot(prediction_df['datetime'], prediction_df[f'iteration_{i}'], color=col, alpha=0.7)
plt.title('Test Interval with All Predictions')
plt.xlabel('Time')
plt.ylabel('Values')

# Create a custom legend entry for the range of colored lines
color_palette_line = mlines.Line2D([], [], color='grey', marker='_', markersize=15, label='Individual Predictions (color range)')
# Get handles and labels from the current plot
handles, labels = plt.gca().get_legend_handles_labels()
# Add the custom legend entry
handles.append(color_palette_line)
# Create the legend with all handles
plt.legend(handles=handles)


# Lower Left Plot: Test Interval with Average Prediction.
plt.subplot(2, 2, 3)
plt.plot(short_interval_datetime, short_interval_original, color='black', label='Original')
plt.axvline(x=last_timestamp_train, color='purple', linestyle='--')
# Plotting the individual predictions
for i in range(n_iter):
    plt.plot(prediction_df['datetime'], prediction_df[f'iteration_{i}'], color='green', alpha=0.7)
# Plotting average prediction
plt.plot(average_all_predictions_df['datetime'], average_all_predictions_df['average_prediction'], color='blue', linewidth=2, label='Average Prediction')
plt.title('Test Interval with Average Prediction')
plt.xlabel('Time')
plt.ylabel('Values')

# Create a custom legend entry for the green lines
green_line = mlines.Line2D([], [], color='green', marker='_', markersize=15, label='Individual Predictions')
# Get handles and labels from the current plot
handles, labels = plt.gca().get_legend_handles_labels()
# Add the custom legend entry
handles.append(green_line)
# Create the legend with all handles
plt.legend(handles=handles)

# Lower Right Plot: Test Interval with Selected Predictions.
plt.subplot(2, 2, 4)
plt.plot(short_interval_datetime, short_interval_original, color='black', label='Original')
plt.axvline(x=last_timestamp_train, color='purple', linestyle='--')
for iteration in selected_iterations:
    plt.plot(selected_predictions_df['datetime'], selected_predictions_df[iteration], color='green', alpha=0.7)
plt.plot(average_selected_predictions_df['datetime'], average_selected_predictions_df['average_selected_prediction'], color='blue', linewidth=2, label='Ensemble Prediction')
plt.title('Test Interval with Selected Predictions')
plt.xlabel('Time')
plt.ylabel('Values')

# Create a custom legend entry for the green lines
green_line = mlines.Line2D([], [], color='green', marker='_', markersize=15, label='Selected Predictions')
# Get handles and labels from the current plot
handles, labels = plt.gca().get_legend_handles_labels()
# Add the custom legend entry
handles.append(green_line)
# Create the legend with all handles
plt.legend(handles=handles)


plt.tight_layout()

# Save the plot as an EPS file
plt.savefig("Overview_ensemble_prediction.eps", format='eps')
plt.show()

