"""
This script is developed as part of the research study 'Potentials and limitations of complexity research in Farming 4.0,'
to be featured in Current Opinion in Environmental Sustainability, 2024 by Mallinger et al. It is aligned with Section 4.3
of the research.

This script applies machine learning to a time series dataset, with the goal of predicting air pollution levels.
The dataset 'Particulate Matter UKAIR 2017' represents hourly air pollution data across Great Britain.
The key idea is to integrate complexity metrics into the prediction process by examining the variance of
second derivatives in a reconstructed phase space. The phase space is reconstructed using a time delay of 1
and an embedding dimension of 100, following the methodology of delay-coordinate embedding.

The machine learning model of choice is a CatBoost Regressor, which is fine-tuned using Bayesian Optimization
to achieve optimal prediction accuracy. The performance of the model is then evaluated on the test dataset.

Complexity in the context of this study is measured by the variance of second derivatives along the phase space
trajectory. We hypothesize that regions of increased complexity correlate with higher prediction errors. This
relationship is explored by categorizing the test data into intervals of low, mid-low, mid-high, and high complexity
based on the computed complexity metric. These intervals are color-coded in the final plot, which also includes
the predictions made by the trained model.

By analyzing the test data, the script identifies patterns between complexity and prediction error, providing
insights into intervals where the model performs well and where it does not. The final output is a visualization
that juxtaposes the actual values with the model's predictions, highlighted with color-coded complexity intervals,
illustrating the relationship between complexity and prediction accuracy.

The script is structured into the following key steps:

1. Import Libraries: Essential Python libraries for data manipulation, machine learning, and phase space embedding are integrated.
2. Set Parameters: Critical parameters for phase space embedding, such as embedding dimensions and time delay, are established.
3. Data Loading and Preprocessing: 'Particulate Matter UKAIR 2017' dataset is prepared, focusing on hourly air pollution data across Great Britain.
4. Phase Space Embedding: The dataset is transformed for time series analysis through phase space embedding.
5. Data Splitting: The transformed dataset is divided into training and testing subsets, essential for machine learning modeling.
6. Model Initialization and Tuning: A CatBoost Regressor is set up and fine-tuned using Bayesian Optimization for optimal prediction accuracy.
7. Model Evaluation: The model's performance on the test set is thoroughly assessed using metrics like the squared error on average for each regime of predictability.
8. Visualization of Model Predictions: Predictive performance is graphically represented, comparing predicted results with actual data and different colourings indicate different complexity regimes.

This script reflects a meticulous focus on applying machine learning techniques within environmental sustainability, underscoring the overarching theme of the research study.
"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 1: Import necessary libraries

# CatBoostRegressor: A machine learning algorithm for regression that is built on decision trees.
from catboost import CatBoostRegressor

# train_test_split: A function to randomly split a dataset into training and testing subsets.
from sklearn.model_selection import train_test_split

# BayesSearchCV: A function to perform hyperparameter tuning using Bayesian optimization.
from skopt import BayesSearchCV

# deepcopy: A function to create a deep copy of data structures, which is a complete copy that doesn't share references with the original.
from copy import deepcopy as dc

# Custom functions from a module dedicated to increasing the predictability of the model.
# This module is specific to the project and should contain project-related helper functions.
from func_increased_predictability import *

# pyplot: A module for creating static, interactive, and animated visualizations in Python.
import matplotlib.pyplot as plt

# mdates: A module for handling date formatting in matplotlib plots.
import matplotlib.dates as mdates

# LineCollection: A class for efficient drawing of lines with shared properties, such as color or line width.
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# numpy: A fundamental package for scientific computing with Python, providing support for arrays and mathematical functions.
import numpy as np

########################################################################################################################
# SETTING PARAMETERS ##################################################################################################
########################################################################################################################
# Step 2: Setting parameters for our machine leanring and analysis approach

# n_iter: The number of iterations for Bayesian Optimization during hyperparameter tuning.
# Increasing n_iter may improve the tuning process at the cost of computational resources and time.
n_iter = 1 # Adjusted to perform a more thorough search over the hyperparameter space.

# prediction_per: Determines the percentage of data to be used as the test set in a train-test split.
# A lower value means more data for training and less for testing, which could lead to better training but potentially overfitting.
prediction_per = 20  # Setting 20% of data for testing, and the rest for training.

# embedding_dimension: The number of past observations (lagged values) to include in the phase space reconstruction.
# This is a critical parameter for capturing the historical trends and patterns in the time series data.
embedding_dimension = 100  # Chosen to include a broader context of past observations in the model.

# time_delay: The time interval between consecutive observations in the phase space.
# It is used to structure the data for the time series model, affecting how patterns are recognized over time.
time_delay = 1  # Set to 1 to use consecutive points without skipping any in the embedding.

# memory_complexity: Represents the number of past points used to compute the complexity of the current point.
# This can affect how sensitive the complexity measure is to recent versus older data.
memory_complexity = 100  # Set to include the last 100 points in the complexity calculations.

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 3: Load and Preprocess Dataset for Phase Space Embedding and calculate complexities

# Loading the dataset
print('Loading Dataset')
X, y = load_dataset()  # Load the 'Particulate Matter UKAIR 2017' dataset
print('Dataset loaded')

# Extract the target time series for phase space embedding
# Here, we're focusing on the 'PM2.5 particulate matter' as our target variable
target_series = X['PM.sub.2.5..sub..particulate.matter..Hourly.measured.'].values
target_series_datetime = X['datetime'].values

# Initialize DataFrame to store complexities with their corresponding timestamps
complexity_df = pd.DataFrame(columns=['datetime', 'complexity'])

# Calculate additional complexity metrics and append to DataFrame
for i in range(memory_complexity, len(target_series)):
    segment = target_series[i-memory_complexity:i]
    complexity = calculate_variance_2nd_derivative(segment)
    timestamp = target_series_datetime[i]
    complexity_df = complexity_df.append({
        'datetime': timestamp,
        'var2der': complexity,
    }, ignore_index=True)

print(target_series)
#original_fractal_dimension = nolds.dfa(target_series[:int((len(target_series)*(1.0 - ((prediction_per * 1.25)/100))))])
original_fractal_dimension = calculate_variance_2nd_derivative(target_series[:int((len(target_series)*(1.0 - ((prediction_per * 1.25)/100))))])

# Transform the dataset using phase space embedding
# This step restructures the data based on the calculated time delay and embedding dimension
X_transformed, y_transformed, X_datetime_transformed, y_datetime_transformed = phase_space_embedding(
    X, embedding_dimension, time_delay)

feature_names = [f'dim_{i+1}' for i in range(embedding_dimension)]

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

# Initialize CatBoost Regressor
# CatBoost is an advanced implementation of gradient boosting, well-suited for time series data
model = CatBoostRegressor(verbose=False)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 6: Model Training and Hyperparameter Optimization

# Set Up Bayesian Optimization with CatBoost
# BayesSearchCV is used for hyperparameter optimization, searching over the specified grid
# It aims to improve model performance by finding the optimal combination of hyperparameters
bocv = BayesSearchCV(model, parameter_grid_cb, n_iter=n_iter, cv=5, verbose=1)

# Training the CatBoost Regressor
print('Training CatBoost Regressor')
# Here, we fit the Bayesian Optimization search with the training data.
# This process not only trains the model but also performs hyperparameter tuning
# to optimize the model's performance based on the provided parameter grid
bocv.fit(X_train, y_train)

# Retrieving the Best Estimator
# After the training and optimization, we extract the best performing model
# along with its parameters and cross-validation score
model = bocv.best_estimator_
best_params = bocv.best_params_
best_cv_score = bocv.best_score_

# Displaying the results of the optimization
# This information is crucial for understanding which parameters worked best
# and how well the model is expected to perform
print("Best Parameters:", best_params)
print("Best CV Score:", best_cv_score)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 7: Model Performance and Complexity Analysis

# Predict and calculate errors for training data
train_predictions = model.predict(X_train)

_ , sq_error_train, _ = calculate_point_errors(y_train, train_predictions)

train_error_df = pd.DataFrame({'squared_error': sq_error_train})
train_error_df['datetime'] = dc(y_datetime_train.values)

# Predict and calculate errors for testing data
test_predictions = model.predict(X_test)

abs_error_test, sq_error_test, _ = calculate_point_errors(y_test, test_predictions)
test_error_df = pd.DataFrame({'squared_error': sq_error_test})
test_error_df['datetime'] = dc(y_datetime_test.values)

# Merge error data with complexity metrics
train_error_df = train_error_df.merge(complexity_df, on='datetime', how='inner')
test_error_df = test_error_df.merge(complexity_df, on='datetime', how='inner')

# Calculate quantiles for each complexity metric
quantiles_var2der = complexity_df['var2der'].quantile([0.25, 0.50, 0.75])

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

# Apply categorization based on quantiles for training and testing data
train_error_df['var2der_quartile'] = train_error_df['var2der'].apply(lambda x: categorize_complexity_intervals(x, quantiles_var2der))
test_error_df['var2der_quartile'] = test_error_df['var2der'].apply(lambda x: categorize_complexity_intervals(x, quantiles_var2der))

# Function to calculate and print mean error statistics per quartile
def print_mean_error_stats_per_quartile(error_df, quartiles, error_types, quantiles):
    for quartile in quartiles:
        print(f"Mean Error Statistics for {quartile}:")
        for error_type in error_types:
            mean_value = error_df[error_df['var2der_quartile'] == quartile][error_type].mean()
            print(f"{error_type} for {quartile}: {mean_value}")
        print("\n")

# Print mean error statistics for Testing Data
print("Testing Data:")
print_mean_error_stats_per_quartile(test_error_df, ['Low', 'Mid-Low', 'Mid-High', 'High'], ['squared_error'], quantiles_var2der)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Step 8: Visualization

# Prepare the dataframe for plotting
plot_df = test_error_df.copy()
y_test_reset = y_test.reset_index(drop=True)
plot_df['actual'] = y_test_reset  # Add actual target values to the dataframe
print('y_test')
print(y_test)
print('Actual')
print(plot_df['actual'])
plot_df['predictions'] = test_predictions

# Sort the dataframe by datetime for proper line plotting
plot_df.sort_values('datetime', inplace=True)

# Convert datetime to matplotlib date numbers
plot_df['datetime_num'] = mdates.date2num(plot_df['datetime'])

# Define the colors for each quartile
color_map = {
    'Low': 'green',
    'Mid-Low': 'yellowgreen',
    'Mid-High': 'orange',
    'High': 'red'
}

# Create segments for actual values
points_actual = np.array([plot_df['datetime_num'], plot_df['actual']]).T.reshape(-1, 1, 2)
segments_actual = np.concatenate([points_actual[:-1], points_actual[1:]], axis=1)

# Create a list of colors for each segment based on quartile
colors_actual = [color_map[quart] for quart in plot_df['var2der_quartile'][:-1]]

# Create a LineCollection for the actual values
lc_actual = LineCollection(segments_actual, colors=colors_actual, linewidth=2)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.add_collection(lc_actual)  # Add the actual values to the plot

# Plotting predictions with a black line
ax.plot(plot_df['datetime'], plot_df['predictions'], color='black', label='Predictions', linewidth=2)

# Formatting date on x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

# Setting the x and y limits
ax.set_xlim(plot_df['datetime'].min(), plot_df['datetime'].max())
ax.set_ylim(plot_df['actual'].min(), plot_df['actual'].max())
print('Actual')
print(plot_df['actual'])
print('Predictions')
print(plot_df['predictions'])

# Additional plot settings
font_size = 12  # Adjust as needed
ax.set_xlabel('Datetime', fontsize=font_size)
ax.set_ylabel('Actual Values', fontsize=font_size)
ax.set_title('Test Data with Color-coded Complexity Intervals and Predictions', fontsize=15)

ax.tick_params(axis='both', which='major', labelsize=font_size)


legend_elements = [
    Line2D([0], [0], color='green', lw=4, label="High Predictability"),
    Line2D([0], [0], color='yellowgreen', lw=4, label="Mid-High Predictability"),
    Line2D([0], [0], color='orange', lw=4, label="Mid-Low Predictability"),
    Line2D([0], [0], color='red', lw=4, label="Low Predictability"),
    Line2D([0], [0], color='black', lw=2, label='Predictions')
]
ax.legend(handles=legend_elements, fontsize=font_size)

# Save the plot in PNG format
plot_filename_png = 'time_series_prediction_complexity_regimes.png'
plt.savefig(plot_filename_png, format='png')

# Save the plot in EPS format
plot_filename_eps = 'time_series_prediction_complexity_regimes.eps'
plt.savefig(plot_filename_eps, format='eps')

# Display the plot
plt.show()