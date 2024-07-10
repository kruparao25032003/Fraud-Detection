import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Read the data
data = pd.read_csv('/kaggle/input/fraudulent-transactions-prediction/Fraud.csv')

# Display basic info
print(data.head())
print(data.info())

# Plot target class distribution
sns.countplot(data=data, x='isFraud')
plt.title('Fraudulent Transactions')
plt.show()

# Prepare data for modeling
data = pd.get_dummies(data, columns=['type'], drop_first=True)
X = data.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
y = data['isFraud']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost Classifier
model = XGBClassifier(random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))
