# Hollywood Box Office Predictor 🎬🍿

A machine learning project that predicts the global box office revenue of a film using strictly pre-release data. This project proves the necessity of non-linear algorithms (like Random Forest) over linear models when forecasting complex, entertainment-industry outcomes.

## Project Architecture
This project has been modularized for clean, production-ready execution.

* `data/IMDb movies.csv`: The raw dataset.
* `data_cleaning.py`: The data engineering pipeline. It handles null values, parses dates, applies one-hot encoding for studios/genres, engineers custom features (Star Power, Franchise Flags), and applies Logarithmic Transformations to normalize skewed financial data.
* `linear_regression.py`: The baseline model script. It calls the cleaning pipeline and runs a standard linear regression to establish a baseline error metric.
* `random_forest.py`: The champion model script. It applies K-Fold Cross Validation and RandomizedSearchCV to tune a robust Random Forest Regressor.
* `presentation.ipynb`: The presentation notebook. It aggregates the pipeline and models into an interactive format, generating visual charts (Feature Importances, MAE/R2 comparisons) for stakeholder review.

## Key Feature Engineering
To capture the reality of the Hollywood business model, the following custom features were engineered:
1. **Star Power:** Cross-referenced actors and directors with an external list of all-time highest-grossing talent.
2. **Franchise IP:** Used Regex to scan titles for numbering or subtitles (e.g., colons) to flag established cinematic universes.

## How to Run Locally
1. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn matplotlib seaborn jupyter`
2. Place the `IMDb movies.csv` inside a `/data` folder.
3. Run the models via terminal:
   * `python linear_regression.py`
   * `python random_forest.py`
4. Or, open `presentation.ipynb` and run all cells for the full visual presentation.