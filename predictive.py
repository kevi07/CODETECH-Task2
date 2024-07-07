import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'data.csv'
car_data = pd.read_csv(file_path)
print("First few rows of the dataset:")
print(car_data.head())
X = car_data.iloc[:, :15]
y = car_data.iloc[:, -1]
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
print(f"Non-numeric columns: {non_numeric_columns}")
X = pd.get_dummies(X, columns=non_numeric_columns)
print("Missing values before handling:")
print(X.isnull().sum())
X = X.fillna(X.mean())
print("Checking for infinite values:")
print(np.isinf(X).sum())
X[~np.isfinite(X)] = np.nan
X = X.fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
plt.figure(figsize=(10, 6))
plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
plt.plot(X_test.iloc[:, 0], y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel(X.columns[0])
plt.ylabel(y.name)
plt.title('Regression Line: Actual vs Predicted')
plt.legend()
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()
