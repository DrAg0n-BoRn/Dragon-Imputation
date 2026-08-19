# Dragon Imputation

A Polars plugin for performing imputation on missing values in DataFrames.

Works on both continuous and categorical variables in numerical form. Categorical variables should be represented as integers (e.g., 0, 1, 2, ...).

MICE imputation uses random decision tree models to predict missing values based on the observed values in the DataFrame. The algorithm iteratively fills in missing values by modeling each variable with missing data as a function of other variables in a round-robin fashion.

## Installation

```bash
pip install dragon-imputation
```

```bash
uv add dragon-imputation
```

## Usage

```python
import polars as pl
from dragon_imputation import mice_impute

# Create a sample DataFrame with missing values
df = pl.DataFrame({
    "A": [1.1, 2.2, None, 4.4],
    "B_cat": [None, 2, 3, 4],
    "C": [1.0, None, 3.0, 4.0]
})

# Perform MICE imputation with 5 iterations
imputed_df = mice_impute(df=df, cat_cols=[1], max_iterations=5)
```

## Arguments

`mice_impute()` function takes the following arguments:
- `df`: A Polars DataFrame containing missing values.
- `cat_cols`: An optional sequence of column indices that are categorical. If None, all columns are treated as continuous.
- `max_iterations`: The maximum number of iterations for the MICE algorithm (default is 30).
- `max_depth`: The maximum depth for the decision tree model used in imputation (default is None for no depth limit).
- `verbose`: The verbosity level for logging (default is 2).
- `seed`: The random seed for reproducibility (default is 42).
- `n_jobs`: The number of threads to use for parallel processing (default is None, which uses all available CPU cores).
