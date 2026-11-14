# House Price Prediction

A regression project that predicts house sale prices from property features with a **Random Forest Regressor**. The notebook covers the full workflow: cleaning, feature engineering, outlier handling, training and evaluation.

## Dataset

`data.csv` has 4,600 house sales from Washington State, USA (May to July 2014), with features such as living area, lot size, bedrooms, bathrooms, floors, condition, year built and location.

## Workflow

1. Load and inspect the data
2. Clean it and drop columns that add no signal (`date`, `year_sold`, `country`)
3. Engineer features: `total_sqft`, `total_rooms`, `house_age`, `is_renovated`; drop the columns they replace
4. Explore with Seaborn / Matplotlib plots
5. Remove outliers with the IQR rule (3,953 of 4,600 rows remain)
6. Split 70% train / 30% test and fit a `RandomForestRegressor`
7. Evaluate on the test set

## Results

| Metric | Value |
|---|---|
| MAE | 81,518.84 |
| RMSE | 118,575.63 |
| R² | 0.682 |

The model explains about two thirds of the variation in price. A typical prediction is off by roughly $80k.

### Caveats

- Outliers (including the target `price`) are removed **before** the train/test split. This makes the test set easier than real-world data, so these scores are probably optimistic. Splitting first and cleaning only the training data would be a fairer evaluation.
- The data covers about ten weeks in one state, so the model should not be used to price houses elsewhere or in other years.
- Some rows in the raw data have a price of 0 (49 rows); the IQR filter removes them.
- Only one model with default hyperparameters is evaluated. Cross-validation, tuning and a baseline (for example linear regression) are natural next steps.

## Run it

Requires Python 3.9+.

```bash
git clone https://github.com/Hammad-Ahmed-hk/House_Price_Prediction_model.git
cd House_Price_Prediction_model
pip install -r requirements.txt
jupyter notebook House_Price_Prediction_ML.ipynb
```

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter

## Author

Hammad Ahmed, [@Hammad-Ahmed-hk](https://github.com/Hammad-Ahmed-hk)
