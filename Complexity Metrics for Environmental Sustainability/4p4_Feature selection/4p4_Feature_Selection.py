"""
This script is developed as part of the research study 'Potentials and limitations of complexity research in Farming 4.0,'
to be featured in Current Opinion in Environmental Sustainability, 2024 by Mallinger et al. It is aligned with Section 4 of the research,
focusing on the application of complexity metrics for feature selection in machine learning, exemplified in the context of pasture prediction.

The script is structured into the following key steps:

1. Import Libraries: Essential Python libraries for data handling, machine learning, and visualization are imported.
2. Data Loading: The 'Pasture Prediction' dataset from OpenML (data_id=339) is loaded for analysis.
3. Mutual Information Computation: Mutual information is calculated to identify the most informative features for pasture prediction.
4. Feature Selection: The top 5 features based on mutual information scores are selected for model training.
5. Data Splitting: The dataset is split into training and testing subsets, which is crucial for evaluating the machine learning model.
6. Model Training: Two RandomForestClassifier models are trained - one with all features and another with selected features.
7. Iterative Model Evaluation: The models are evaluated across 100 iterations to assess their stability and performance consistency.
8. Visualization of Results: Classification accuracies for each iteration are visually compared through a line plot, and the results are saved as an EPS file.
9. Result Storage: The average accuracies and iteration-wise scores are stored in a JSON file for further analysis.

This script is a key component of our research, demonstrating the effective use of feature selection and iterative evaluation in enhancing the predictive accuracy of machine learning models in the agricultural domain.
"""

# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 2: Data Loading
print("Loading the dataset...")
dataset = fetch_openml(data_id=339)

X = dataset.data
y = dataset.target

if isinstance(X, pd.DataFrame):
    X = X.select_dtypes(include=[np.number])  # select only numeric features
if isinstance(y, pd.Series) and y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)  # encode non-numeric labels
X = np.array(X)  # Ensure X is a NumPy array


# Initialize variables to accumulate scores
scores_all = []
scores_selected = []

# Step 7: Iterative Model Evaluation
for i in range(100):
    # Step 5: Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Step 3: Mutual Information Computation
    mi_scores = mutual_info_classif(X_train, y_train)
    # Step 4: Feature Selection
    top_features = np.argsort(mi_scores)[::-1][:5]

    # Step 6: Model Training with all features using Decision Tree
    model_all = DecisionTreeClassifier(random_state=42)
    model_all.fit(X_train, y_train)
    predictions_all = model_all.predict(X_test)
    score_all = accuracy_score(y_test, predictions_all)
    scores_all.append(score_all)
    print(f"Iteration {i + 1} - Accuracy with all features: {score_all:.4f}")

    # Step 6: Model Training with selected features using Decision Tree
    X_train_selected = X_train[:, top_features]
    X_test_selected = X_test[:, top_features]
    model_selected = DecisionTreeClassifier(random_state=42)
    model_selected.fit(X_train_selected, y_train)
    predictions_selected = model_selected.predict(X_test_selected)
    score_selected = accuracy_score(y_test, predictions_selected)
    scores_selected.append(score_selected)
    print(f"Iteration {i + 1} - Accuracy with selected features: {score_selected:.4f}")

# Step 8: Visualization of Results
plt.figure(figsize=(15, 8))  # Increased figure size
plt.plot(range(1, 101), scores_all, label='All Features', linestyle='-', color='blue')
plt.plot(range(1, 101), scores_selected, label='Selected Features based on Mutual Information', linestyle='-', color='orange')
plt.title('Classification Accuracies over 100 Iterations: All Features vs. MI selected Features', fontsize=14)
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
    'All Features': scores_all,
    'MI Selected Features': scores_selected
})

# Plotting boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=boxplot_data)

# Increase the font size of plot description (title, x-label, y-label, and ticks)
plt.title('Boxplot of Classification Accuracies: All Features vs. MI Selected Features', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)

# You can also increase the font size of the tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('classification_accuracies_boxplot.eps', format='eps')  # Save the plot as EPS
plt.show()


# Step 9: Result Storage
avg_score_all = np.mean(scores_all)
avg_score_selected = np.mean(scores_selected)
print("\nAverage accuracy with all features:", avg_score_all)
print("Average accuracy with selected features:", avg_score_selected)

results = {
    "Average Accuracy All Features": avg_score_all,
    "Average Accuracy Selected Features": avg_score_selected,
    "Scores All Features": scores_all,
    "Scores Selected Features": scores_selected
}
with open('accuracy_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to JSON file and plot saved as EPS.")
