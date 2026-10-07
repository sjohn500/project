# Day 1 — EDA on House Prices Dataset

## What I did
- Set up Python venv with pandas, numpy, sklearn, matplotlib, seaborn, streamlit
- Configured Kaggle CLI and downloaded the House Prices dataset
- Ran first EDA on train.csv (1460 rows, 81 columns)

## What I learned

### Dataset basics
- 1460 rows = houses, 81 columns = 80 features + 1 target (SalePrice)
- This is a supervised regression problem: predict a continuous number

### Skewness
- SalePrice mean (~181k) > median (~163k) → right-skewed distribution
- Skewed targets hurt linear models → will log-transform in Day 2

### Missing values
- 19 columns have missing data
- Missingness is meaningful, not random — e.g., PoolQC NA = "no pool"
- Need per-column strategy, not blanket dropna()

### Correlations
- OverallQual (r=0.79) is the strongest predictor of SalePrice
- GrLivArea (r=0.71), GarageCars (r=0.64) next
- Correlation ranges from -1 to 1; 0 = no relationship

### EDA mindset
- Always explore data BEFORE modeling
- Check: shape, target distribution, missing values, correlations

## What confused me
[write 1-2 honest lines — e.g., "why is correlation important?" or "when do I use log?"]

## Tomorrow's plan
- Log-transform SalePrice
- Handle missing values per column type
- Create engineered features (TotalSF, HouseAge, etc.)
- Encode categorical columns
