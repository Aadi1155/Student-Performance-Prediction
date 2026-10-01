
# ============================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# ============================================================

# -------------------------
# STEP 1: IMPORT LIBRARIES
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


from sklearn.metrics import mean_squared_error, r2_score
   

# -------------------------
# STEP 2: LOAD DATASET
# -------------------------

data = pd.read_csv("student-mat.csv", sep=";")


# -------------------------
# STEP 3: UNDERSTAND DATA
# -------------------------

print("========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== COLUMN NAMES ==========")
print(data.columns.tolist())


# -------------------------
# STEP 4: MISSING VALUES
# -------------------------

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())


# -------------------------
# STEP 5: DUPLICATES
# -------------------------

print("\n========== DUPLICATE ROWS ==========")
print(data.duplicated().sum())

data = data.drop_duplicates()


# -------------------------
# STEP 6: DEFINE X AND y
# -------------------------

X = data.drop("G3", axis=1)
y = data["G3"]

print("\n========== INPUT SHAPE ==========")
print(X.shape)

print("\n========== TARGET SHAPE ==========")
print(y.shape)


# -------------------------
# STEP 7: IDENTIFY COLUMNS
# -------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

print("\n========== NUMERICAL FEATURES ==========")
print(list(numerical_features))

print("\n========== CATEGORICAL FEATURES ==========")
print(list(categorical_features))


# -------------------------
# STEP 8: PREPROCESSING
# -------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# -------------------------
# STEP 9: EDA
# -------------------------

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())


# Study time vs final grade
plt.figure(figsize=(8, 5))
plt.scatter(data["studytime"], data["G3"])
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.title("Study Time vs Final Grade")
plt.show()


# G1 vs G3
plt.figure(figsize=(8, 5))
plt.scatter(data["G1"], data["G3"])
plt.xlabel("First Period Grade (G1)")
plt.ylabel("Final Grade (G3)")
plt.title("G1 vs Final Grade")
plt.show()


# G2 vs G3
plt.figure(figsize=(8, 5))
plt.scatter(data["G2"], data["G3"])
plt.xlabel("Second Period Grade (G2)")
plt.ylabel("Final Grade (G3)")
plt.title("G2 vs Final Grade")
plt.show()


# Absences vs G3
plt.figure(figsize=(8, 5))
plt.scatter(data["absences"], data["G3"])
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.title("Absences vs Final Grade")
plt.show()


# Correlation
correlation = data.select_dtypes(
    include=["int64", "float64"]
).corr()

print("\n========== CORRELATION WITH G3 ==========")
print(correlation["G3"].sort_values(ascending=False))


# -------------------------
# STEP 10: TRAIN/TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -------------------------
# STEP 11: LINEAR REGRESSION
# -------------------------

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# -------------------------
# STEP 12: DECISION TREE
# -------------------------

tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)


# -------------------------
# STEP 13: RANDOM FOREST
# -------------------------

forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    ))
])

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)


# -------------------------
# STEP 14: EVALUATION
# -------------------------

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

tree_rmse = np.sqrt(
    sklearn.metrics.mean_squared_error(y_test, tree_predictions)
)

forest_rmse = np.sqrt(
    sklearn.metrics.mean_squared_error(y_test, forest_predictions)
)


linear_r2 = sklearn.metrics.r2_score(y_test, linear_predictions)
tree_r2 = sklearn.metrics.r2_score(y_test, tree_predictions)
forest_r2 = sklearn.metrics.r2_score(y_test, forest_predictions)


# -------------------------
# STEP 15: DISPLAY RESULTS
# -------------------------

print("\n====================================")
print("         MODEL RESULTS")
print("====================================")

print("\nLinear Regression")
print("RMSE:", round(linear_rmse, 2))
print("R² Score:", round(linear_r2, 2))

print("\nDecision Tree")
print("RMSE:", round(tree_rmse, 2))
print("R² Score:", round(tree_r2, 2))

print("\nRandom Forest")
print("RMSE:", round(forest_rmse, 2))
print("R² Score:", round(forest_r2, 2))


# -------------------------
# STEP 16: ACTUAL VS PREDICTED
# -------------------------

results = pd.DataFrame({
    "Actual Grade": y_test.values,
    "Predicted Grade": forest_predictions
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(results.head(10))


# -------------------------
# STEP 17: PREDICTION GRAPH
# -------------------------

plt.figure(figsize=(8, 5))

plt.scatter(y_test, forest_predictions)

plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Final Grade")

plt.show()