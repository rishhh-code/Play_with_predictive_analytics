# 1. Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score
import joblib
import os

# 2. Load and clean the dataset
dataset = pd.read_csv('data/house_data.csv')
dataset = dataset.dropna()

# 3. Create dummy variables for 'arrondissement'
dummies = pd.get_dummies(dataset['arrondissement'].astype(str), prefix='arrondissement')
dataset = pd.concat([dummies, dataset[['price', 'surface']]], axis=1)

# 4. Feature matrix (X) and target vector (y)
X = dataset.drop('price', axis=1)
y = dataset['price']

# 5. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 6. Add intercept column for OLS
X_train_ols = sm.add_constant(X_train)

# 7. Backward Elimination
X_opt = X_train_ols.copy()
cols = list(X_opt.columns)
while True:
    model = sm.OLS(y_train.astype(float), X_opt.astype(float)).fit()
    p_values = model.pvalues
    max_p = p_values.max()
    if max_p > 0.05:
        excluded_feature = p_values.idxmax()
        cols.remove(excluded_feature)
        X_opt = X_train_ols[cols]
    else:
        break

# 8. Final selected features
print("Selected features after Backward Elimination:")
print(cols)

# 9. Train the final model using sklearn
# Exclude 'const' from training with sklearn
final_features = cols[1:]
X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

regressor = LinearRegression()
regressor.fit(X_train_final, y_train)

# 10. Predict and evaluate
y_pred = regressor.predict(X_test_final)
r2 = r2_score(y_test, y_pred)

print("\nModel Parameters:")
print("Intercept:", regressor.intercept_)
print("Coefficients:", regressor.coef_)
print("R² Score on Test Data:", r2)

# 11. Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(regressor, "models/mlr1.pkl")
print("\n✅ Model saved as 'models/mlr1.pkl'")

# 12. Save selected features to a txt file (for prediction UI to use)
with open("models/mlr1_features.txt", "w") as f:
    for feature in final_features:
        f.write(feature + "\n")
print("✅ Features saved to 'models/mlr1_features.txt'")
