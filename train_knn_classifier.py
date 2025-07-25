
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("squat_features.csv")
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

y_pred = knn.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
