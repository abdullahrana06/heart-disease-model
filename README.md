# heart-disease-model
A machine learning project to predict heart disease using health data.
~step by step process-

1) Setup & Imports
- Installs and imports: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- Sets a random seed for reproducibility.

2) Load the Data
- Reads `heart.csv` into a DataFrame.
- Shows the first few rows and basic info (`.head()`, `.info()`, `.describe()`).

3) Clean & Prepare
- Checks for missing values and handles them (imputation if needed).
- Splits **features** (`X`) and **target** (`y`).
- (Optional) Applies scaling for models that benefit (e.g., Logistic Regression).

4) Exploratory Data Analysis (EDA)
- Visualizes distributions and correlations to understand relationships.
- Simple plots to spot trends and potential data leakage.

5) Train/Test Split
- Splits the data (e.g., 80% train / 20% test) using `train_test_split`.

6) Train Baseline Model — Logistic Regression
- Fits a logistic regression model (`LogisticRegression(max_iter=1000)`).
- Predicts on the test set.

7) Try Another Model — Decision Tree
- Fits a `DecisionTreeClassifier` (with simple, readable hyperparameters).
- Compares results to the baseline model.

8) Evaluate Models
- **Accuracy**: `accuracy_score(y_test, y_pred)`
- **ROC–AUC**: `roc_auc_score(y_test, y_proba)` + ROC curve
- **Confusion Matrix**: clear visual heatmap

9) Feature Importance
- For Logistic Regression: uses model coefficients (after scaling).
- For Decision Tree: uses `.feature_importances_`.
- Displays a sorted bar chart of most influential features.
