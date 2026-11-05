from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

# Load dataset
file_path = Path(__file__).resolve().parent / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
if not file_path.exists():
    raise FileNotFoundError("Place ObesityDataSet_raw_and_data_sinthetic.csv in the data folder.")
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Separate input and target
target = "NObeyesdad"
X = df.drop(columns=[target])
y = df[target]

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(),
    "MLP Neural Network": MLPClassifier(
        max_iter=500, random_state=42
    )
}

results = []
trained_models = {}

# Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
    })

    trained_models[name] = pipeline

# Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1", ascending=False
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))

# Select best model using Macro F1
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = best_model.predict(X_test)

print("\nBest Model:", best_name)
results_df.to_csv(Path(__file__).resolve().parent / "results" / "model_comparison.csv", index=False)
joblib.dump(best_model, Path(__file__).resolve().parent / "results" / "best_obesity_model.pkl")
print("\nSaved model comparison and best model in results/.")

print("\nClassification Report:")
print(classification_report(y_test, best_predictions, zero_division=0))

# Plot comparison
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "Macro F1"]
].plot(kind="bar", figsize=(12, 6))

plt.title("Obesity Prediction - Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "results" / "obesity_model_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig(Path(__file__).resolve().parent / "results" / "obesity_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, best_predictions, xticks_rotation="vertical"
)
plt.title("Confusion Matrix - Best Obesity Model")
plt.tight_layout()
plt.show()