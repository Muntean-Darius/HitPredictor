import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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


df = pd.read_csv('IMDb movies.csv', low_memory=False)

columns_to_keep = ['title', 'date_published', 'genre', 'budget', 'worlwide_gross_income', 'metascore', 'production_company', 'actors', 'director']
df = df[columns_to_keep]

df = df.dropna(subset=['budget', 'worlwide_gross_income', 'metascore'])


df = df[df['budget'].str.contains(r'\$', na=False)]
df = df[df['worlwide_gross_income'].str.contains(r'\$', na=False)]

df['budget'] = df['budget'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)
df['worlwide_gross_income'] = df['worlwide_gross_income'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)


df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')
df = df.dropna(subset=['date_published'])
df['release_month'] = df['date_published'].dt.month
df = df.drop('date_published', axis=1)


top_10_studios = df['production_company'].value_counts().nlargest(10).index
df['production_company'] = df['production_company'].apply(lambda x: x if x in top_10_studios else 'Other')


studio_dummies = pd.get_dummies(df['production_company'], prefix='studio')
genre_dummies = df['genre'].str.get_dummies(sep=', ')

df = pd.concat([df, studio_dummies, genre_dummies], axis=1)
df = df.drop(['production_company', 'genre'], axis=1)


df['star_actor_count'] = df['actors'].apply(count_star_power)
df['has_star_director'] = df['director'].apply(check_director_power)

df = df.drop(['actors', 'director'], axis=1)

print(f"Data cleaned successfully! You now have {len(df)} movies ready for machine learning.")

X = df.drop(['worlwide_gross_income', 'title'], axis=1)
y = df['worlwide_gross_income']


print("\nHere is a preview of your cleaned data matrix:")
print(df.head())


print("Splitting data into 80% Training and 20% Testing...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} movies. Testing on {len(X_test)} movies.")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Making predictions on the test data...")
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Results ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared Score: {r2:.4f}")