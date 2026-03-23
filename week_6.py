import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X, y = make_classification(n_samples=5000, n_features=12, n_informative=8, n_redundant=2, n_classes=2, random_state=42)

columns = ["Tenure","MonthlyCharges","TotalCharges","ContractType","InternetService",
           "TechSupport","OnlineSecurity","StreamingTV","PaymentMethod","SeniorCitizen",
           "Dependents","Partner"]

df = pd.DataFrame(X, columns=columns)
df["Churn"] = y

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

