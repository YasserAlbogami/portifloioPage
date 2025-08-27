import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create dummy dataset
X = np.random.rand(100, 3)   # 100 samples, 3 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # simple rule: if sum of first 2 > 1 → class 1 else 0

# 2. Put in pandas DataFrame
df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
df["label"] = y

print("Sample data:")
print(df.head())

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df[["f1", "f2", "f3"]], df["label"], test_size=0.2, random_state=42
)

# 4. Train logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 5. Predict and check accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
