import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from data_cleaning import get_cleaned_data # Importing your custom pipeline

# 1. Get the data
X_train, X_test, y_train, y_test, features = get_cleaned_data()

# 2. Train the baseline model
print("--- Running Linear Regression Baseline ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 3. Predict and reverse log transformation
lr_pred_log = lr_model.predict(X_test)
lr_pred_dollars = np.expm1(lr_pred_log)
y_test_dollars = np.expm1(y_test)

# 4. Evaluate
lr_mae = mean_absolute_error(y_test_dollars, lr_pred_dollars)
lr_r2 = r2_score(y_test_dollars, lr_pred_dollars)

print(f"MAE: ${lr_mae:,.2f}")
print(f"R-Squared: {lr_r2:.4f}\n")