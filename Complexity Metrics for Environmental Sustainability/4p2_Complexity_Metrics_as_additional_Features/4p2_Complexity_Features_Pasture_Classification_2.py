"""
This script is part of a research project exploring the effectiveness of complexity features in improving the predictability of machine learning models. The script demonstrates the process of loading the Pasture Production dataset, optionally extracting additional complexity features, and utilizing a variety of machine learning classifiers through LazyPredict for classification.

The script performs the following steps:
1. Imports necessary libraries for data handling, machine learning, complexity feature calculations, and model evaluation.
2. Loads the 'Pasture Production' dataset from OpenML, which contains biophysical indicators from grazed North Island hill country areas for predicting pasture production.
3. Optionally adds complexity features to the dataset (if enabled) to enhance the feature space. This step includes metrics like Sample Entropy, Approximate Entropy, and Spectral Entropy, capturing various aspects of irregularity and unpredictability within the data.
4. Model Training: Two Extra Trees Classifier models are trained - one with the original features and another with original features + computed complexity features.
5. Iterative Model Evaluation: The two models (using only the original features and using onlriginal features + computed complexity features) are evaluated across 100 iterations to assess their stability and performance consistency.
6. Visualization of Results: Classification accuracies for each iteration are visually compared through a line plot and a box plot, and the results are saved as an EPS file.
7. Result Storage: The average accuracies and iteration-wise scores are stored in a JSON file for further analysis.

The inclusion of complexity features is a key aspect of this research, as they potentially add valuable information to the models, leading to enhanced predictive accuracy. This approach allows for a rapid assessment of various models' performances on the dataset, facilitating more informed decisions in the model selection process.

"""


# Step 1: Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from lazypredict.Supervised import LazyClassifier
import numpy as np
np.random.seed(42)
import pandas as pd
import antropy as ant
import json
from copy import deepcopy as dc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Step 2: Load Data
def load_dataset(data_id=339):
    """
    Loads the 'Pasture Production' dataset from OpenML by its dataset ID.

    The Pasture Production dataset, curated by Dave Barker from AgResearch Grasslands, Palmerston North, New Zealand, is aimed at predicting pasture production using a variety of biophysical factors. It includes data from areas of grazed North Island hill country with different management histories (fertilizer application and stocking rate) from 1973 to 1994. The dataset is subdivided into 36 paddocks with 19 selected biophysical indicators, encompassing vegetation, soil chemical, physical, biological, and soil water variables.

    The objective is to use these biophysical indicators to predict pasture production, which is crucial for effective agricultural management.

    Attribute Information:
    The dataset includes attributes such as fertiliser usage, slope, aspect deviation from the north-west, soil chemical properties (OlsenP, MinN, TS, Ca-Mg ratio), soil physical properties (LOM, KUnSat, OM, Air-Perm, Porosity), biological properties (NFIX-mean, earthworms), vegetation properties (HFRG-pct-mean, legume-yield, OSPP-pct-mean), soil water content (Jan-Mar-mean-TDR), annual mean runoff, root surface area, leaf phosphorus content, and pasture production categorisation.

    Dataset Author: Dave Barker
    Source: OpenML
    Please cite:
    Barker, D. (Year). Pasture Production Data. AgResearch Grasslands, Palmerston North, New Zealand.

    Access Link:
    - OpenML: https://www.openml.org/d/339

    Accessed on: 2023-12-27

    Parameters:
    - data_id (int): The OpenML ID for the dataset. Default is 339 for the Pasture Production dataset.

    Returns:
    - X (DataFrame): Features of the dataset.
    - y (Series): Target variable.
    """

    data_sk = fetch_openml(data_id=data_id, as_frame=True)

    X = data_sk.data
    y = data_sk.target

    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include=[np.number])  # select only numeric features

    if isinstance(y, pd.Series) and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)  # encode non-numeric labels

    return X, y


#Step 3: Potentially add complexity metrics
def calculate_complexity_features(X, use_sample_entropy=True, use_approx_entropy=True, use_spectral_entropy=False):
    """
    This function calculates and adds various complexity features to the dataset to enhance
    its predictability in machine learning models. These features include Sample Entropy,
    Approximate Entropy, Detrended Fluctuation Analysis (DFA), and the Hurst Exponent.
    Each of these metrics captures different aspects of irregularity or unpredictability
    within time-series data.

    - Sample Entropy: A measure of the regularity and unpredictability in of data.
      Lower values indicate more self-similarity and regularity, while higher values
      suggest more randomness. It is useful for analyzing the complexity of physiological
      signals and other time-series data.

    - Approximate Entropy: Quantifies the unpredictability of fluctuations in data.
      It is similar to Sample Entropy but less sensitive to small fluctuations. This metric
      is particularly helpful in distinguishing time-series data with subtle dynamic changes.

    - Spectral Entropy: Represents the entropy of the power spectrum of data. It
      quantifies the regularity and randomness in the signal's frequency domain.

    The `antropy` package offers other complexity metrics as well, such as Lempel-Ziv Complexity or
    Permutation Entropy. These metrics can also be explored for additional
    insights into the data. Many metrics are specifically designed to deal with time series data.
    However, these metrics can be applied to arbitrary numerical data and can potentially increase
    a model's accuracy.

    Parameters:
    - X (DataFrame): The input dataset with features as columns.
    - use_sample_entropy (bool): Flag to compute and add Sample Entropy.
    - use_approx_entropy (bool): Flag to compute and add Approximate Entropy.
    - use_spectral_entropy (bool): Flag to compute and add Spectral Entropy.

    Returns:
    - X (DataFrame): The modified dataset with added complexity features.
    """

    print('Calculating Complexities:')
    X_original = dc(X)  # Deep copy of the original DataFrame

    if use_sample_entropy and 'sample_entropy' not in X.columns:
        print('Sample Entropy')
        X_original = X_original.astype(np.float32)
        sample_entropy = X_original.apply(lambda row: ant.sample_entropy(row), axis=1)
        X['sample_entropy'] = sample_entropy

    if use_approx_entropy and 'approx_entropy' not in X.columns:
        print('Approximate Entropy')
        approximate_entropy = X_original.apply(lambda row: ant.app_entropy(row), axis=1)
        X['approximate_entropy'] = approximate_entropy

    if use_spectral_entropy and 'spectral_entropy' not in X.columns:
        print('Spectral Entropy')
        spectral_entropy = X_original.apply(lambda row: ant.spectral_entropy(row, sf=8.0), axis=1)
        X['spectral_entropy'] = spectral_entropy

    return dc(X)

# set to true if you want to include complexity features in the code
use_complexity_features = True


print('Loading Dataset')
X, y = load_dataset()
print('Dataset loaded')

if use_complexity_features:
    X = calculate_complexity_features(X)

all_feature_names = list(X.columns) # Store all feature names

columns_to_delete = ['sample_entropy', 'approximate_entropy', 'spectral_entropy']

# Delete specified column names from feature_names using list comprehension
feature_names = [col for col in all_feature_names if col not in columns_to_delete] # Store original feature names
print(feature_names)



# Step 5: Iterative Model Evaluation
# Initialize variables to accumulate scores
scores_all = []
scores_orig = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Step 4: Model Training with added complexity features using Extra Trees Classifier
    model_all = ExtraTreesClassifier(random_state=42)
    model_all.fit(X_train, y_train)
    predictions_all = model_all.predict(X_test)
    score_all = accuracy_score(y_test, predictions_all)
    scores_all.append(score_all)
    print(f"Iteration {i + 1} - Accuracy with original + additional complexity features: {score_all:.4f}")

    # Step 4: Model Training with original features using Extra Trees Classifier
    X_train_orig = X_train.loc[:, feature_names]
    X_test_orig = X_test.loc[:, feature_names]
    model_orig = ExtraTreesClassifier(random_state=42)
    model_orig.fit(X_train_orig, y_train)
    predictions_orig = model_orig.predict(X_test_orig)
    score_orig = accuracy_score(y_test, predictions_orig)
    scores_orig.append(score_orig)
    print(f"Iteration {i + 1} - Accuracy with original features: {score_orig:.4f}")


# Step 6: Visualization of Results
plt.figure(figsize=(15, 8))  # Increased figure size
plt.plot(range(1, 101), scores_all, label='Original + Additional Complexity Features', linestyle='-', color='blue')
plt.plot(range(1, 101), scores_orig, label='Original Features', linestyle='-', color='orange')
plt.title('Classification Accuracies over 100 Iterations: Original + Additional Complexity Features vs. Original Features', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(range(0, 101, 10))  # Show every 10th iteration
plt.legend(fontsize=12)
plt.grid(True)  # Adding a grid
plt.savefig('classification_accuracies.eps', format='eps')  # Save the plot as EPS
plt.show()



# Creating a DataFrame for the boxplot data
boxplot_data = pd.DataFrame({
    'Original + Complexity Features': scores_all,
    'Original Features': scores_orig
})
# Plotting boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=boxplot_data)

# Increase the font size of plot description (title, x-label, y-label, and ticks)
plt.title('Boxplot of Classification Accuracies: Original + Additional Complexity Features vs. Original Features', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)

# You can also increase the font size of the tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.savefig('classification_accuracies_boxplot.eps', format='eps')  # Save the plot as EPS
plt.show()




# Step 7: Result Storage
avg_score_orig = np.mean(scores_orig)
avg_score_all = np.mean(scores_all)
print("\nAverage accuracy with original features:", avg_score_orig)
print("Average accuracy with origial + complexity features:", avg_score_all)

results = {
    "Average Accuracy Original Features": avg_score_orig,
    "Average Accuracy Original + Complexity Features": avg_score_all,
    "Scores Original Features": scores_orig,
    "Scores Original + Complexity Features": scores_all
}
with open('accuracy_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to JSON file and plot saved as EPS.")
