import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV # Model evaluation and tuning tools

# --- Feature Engineering Setup ---
a_list_actors = [
    'Tom Cruise', 'Robert Downey Jr.', 'Scarlett Johansson', 'Leonardo DiCaprio',
    'Dwayne Johnson', 'Chris Hemsworth', 'Chris Evans', 'Margot Robbie',
    'Tom Hanks', 'Samuel L. Jackson', 'Brad Pitt', 'Zoe Saldana',
    'Will Smith', 'Harrison Ford', 'Jennifer Lawrence', 'Chris Pratt',
    'Johnny Depp', 'Tom Holland', 'Denzel Washington', 'Angelina Jolie',
    'Vin Diesel', 'Bradley Cooper', 'Mark Ruffalo', 'Emma Watson'
]

a_list_directors = [
    'Steven Spielberg', 'James Cameron', 'Christopher Nolan', 'Peter Jackson',
    'Michael Bay', 'J.J. Abrams', 'Ridley Scott', 'Quentin Tarantino',
    'Martin Scorsese', 'Denis Villeneuve', 'David Fincher', 'Zack Snyder'
]

def count_star_power(actor_string):
    if pd.isna(actor_string):
        return 0
    movie_actors = [actor.strip() for actor in actor_string.split(',')]
    star_count = 0
    for actor in movie_actors:
        if actor in a_list_actors:
            star_count += 1
    return star_count

def check_director_power(director_string):
    if pd.isna(director_string):
        return 0
    if director_string.strip() in a_list_directors:
        return 1
    return 0

# --- Data Loading and Cleaning ---
df = pd.read_csv('IMDb movies.csv', low_memory=False)

columns_to_keep = ['title', 'date_published', 'genre', 'budget', 'worlwide_gross_income', 'metascore', 'production_company','actors', 'director']
df = df[columns_to_keep]

df = df.dropna(subset=['budget', 'worlwide_gross_income', 'metascore'])

df = df[df['budget'].str.contains(r'\$', na=False)]
df = df[df['worlwide_gross_income'].str.contains(r'\$', na=False)]

# Convert text to standard float numbers
df['budget'] = df['budget'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)
df['worlwide_gross_income'] = df['worlwide_gross_income'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)

# Logarithmic Transformation
# Applies np.log1p (log base e + 1) to normalize skewed data
df['budget'] = np.log1p(df['budget'])
df['worlwide_gross_income'] = np.log1p(df['worlwide_gross_income'])

# Date parsing
df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')
df = df.dropna(subset=['date_published'])
df['release_month'] = df['date_published'].dt.month
df = df.drop('date_published', axis=1)

# Taming Studios and Genres
top_10_studios = df['production_company'].value_counts().nlargest(10).index
df['production_company'] = df['production_company'].apply(lambda x: x if x in top_10_studios else 'Other')

studio_dummies = pd.get_dummies(df['production_company'], prefix='studio')
genre_dummies = df['genre'].str.get_dummies(sep=', ')

df = pd.concat([df, studio_dummies, genre_dummies], axis=1)
df = df.drop(['production_company', 'genre'], axis=1)

# Applying Star Power
df['star_actor_count'] = df['actors'].apply(count_star_power)
df['has_star_director'] = df['director'].apply(check_director_power)
df = df.drop(['actors', 'director'], axis=1)

# Define X and y
X = df.drop(['worlwide_gross_income', 'title'], axis=1)
y = df['worlwide_gross_income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Machine Learning ---

# K-Fold Cross Validation
baseline_rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(baseline_rf, X_train, y_train, cv=5, scoring='r2')
print(f"Average CV R-squared Score: {cv_scores.mean():.4f}")

# Hyperparameter Tuning
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
print(f"Best settings found: {tuner.best_params_}")

# --- Final Evaluation ---
y_pred_log = best_rf_model.predict(X_test)

# Reverse the logarithmic transformation
y_test_dollars = np.expm1(y_test)
y_pred_dollars = np.expm1(y_pred_log)

# Calculate final scores using real dollars
mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
r2 = r2_score(y_test_dollars, y_pred_dollars)

print("\n--- Final Model Results ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared Score: {r2:.4f}")