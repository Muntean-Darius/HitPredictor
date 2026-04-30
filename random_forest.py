import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from data_cleaning import get_cleaned_data # Importing your custom pipeline

# 1. Get the data
X_train, X_test, y_train, y_test, features = get_cleaned_data()

print("--- Running Advanced Random Forest ---")

# 2. Cross Validation Stress Test
baseline_rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(baseline_rf, X_train, y_train, cv=5, scoring='r2')
print(f"5-Fold CV Average R-squared: {cv_scores.mean():.4f}")

# 3. Hyperparameter Tuning
print("\nTuning Hyperparameters (n_jobs=-1 using all cores)...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

tuner = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    random_state=42,
    n_jobs=-1
)
tuner.fit(X_train, y_train)

best_rf_model = tuner.best_estimator_
print(f"Optimal settings found: {tuner.best_params_}")

# 4. Final Predictions and Math Reversal
y_pred_log = best_rf_model.predict(X_test)
y_pred_dollars = np.expm1(y_pred_log)
y_test_dollars = np.expm1(y_test)

# 5. Evaluate
rf_mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
rf_r2 = r2_score(y_test_dollars, y_pred_dollars)

print("\n--- Final Tuned Results ---")
print(f"MAE: ${rf_mae:,.2f}")
print(f"R-Squared: {rf_r2:.4f}")