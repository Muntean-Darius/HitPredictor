import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# --- Feature Engineering Lists ---
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


def get_cleaned_data(filepath='data/IMDb movies.csv'):
    """Loads, cleans, and splits the dataset for machine learning."""
    print("Initializing Data Pipeline...")
    df = pd.read_csv(filepath, low_memory=False)

    columns_to_keep = ['title', 'date_published', 'genre', 'budget', 'worlwide_gross_income', 'metascore',
                       'production_company', 'actors', 'director']
    df = df[columns_to_keep]

    df = df.dropna(subset=['budget', 'worlwide_gross_income', 'metascore'])
    df = df[df['budget'].str.contains(r'\$', na=False)]
    df = df[df['worlwide_gross_income'].str.contains(r'\$', na=False)]

    # Currency conversion
    df['budget'] = df['budget'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(
        float)
    df['worlwide_gross_income'] = df['worlwide_gross_income'].str.replace('$', '', regex=False).str.replace(',', '',
                                                                                                            regex=False).str.strip().astype(
        float)

    # Logarithmic Transformation
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

    # Feature Engineering
    df['star_actor_count'] = df['actors'].apply(count_star_power)
    df['has_star_director'] = df['director'].apply(check_director_power)
    df['is_franchise'] = df['title'].apply(lambda x: 1 if re.search(r'(:|\d)', str(x)) else 0)

    df = df.drop(['actors', 'director', 'title'], axis=1)

    # Matrix Split
    X = df.drop(['worlwide_gross_income'], axis=1)
    y = df['worlwide_gross_income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Pipeline Complete: Generated matrices for {len(df)} movies.\n")
    return X_train, X_test, y_train, y_test, X.columns