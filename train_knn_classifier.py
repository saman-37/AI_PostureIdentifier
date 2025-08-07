import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load the dataset
df = pd.read_csv("squat_features.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 2. Split the dataset (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Build pipeline (scaling + KNN)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling is essential!
    ('knn', KNeighborsClassifier())
])

# 4. Define hyperparameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7],  # Search for optimal number of neighbors
    'knn__weights': ['uniform', 'distance'],  # Weighting strategy
    'knn__metric': ['euclidean', 'manhattan']  # Distance metric
}

# 5. Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=1
)
grid_search.fit(X_train, y_train)

# 6. Save the best model
best_model = grid_search.best_estimator_
with open("knn_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# 7. Evaluate model performance
y_pred = best_model.predict(X_test)
print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- After you have your best KNN model (e.g., `best_knn`) and your scaled test data ---
y_pred = best_model.predict(X_test)

# Build confusion matrix
labels = sorted(set(y_test) | set(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(4,4), dpi=300)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, values_format="d")
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")  # Saves for your poster
plt.show()
