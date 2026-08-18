# Polars Imputation Plugin

A Polars plugin for performing imputation on missing values in DataFrames.

MICE imputation uses a decision tree model to predict missing values based on the observed values in the DataFrame. The algorithm iteratively fills in missing values by modeling each variable with missing data as a function of other variables in a round-robin fashion.

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
    "A": [1, 2, None, 4],
    "B": [None, 2, 3, 4],
    "C": [1, None, 3, 4]
})

# Perform MICE imputation with 5 iterations
imputed_df = mice_impute(df, 5)
```

## Arguments

`mice_impute()` function takes the following arguments:
- `pydf`: A Polars DataFrame containing missing values.
- `max_iterations`: The maximum number of iterations for the MICE algorithm (default is 10).
- `max_depth`: The maximum depth for the decision tree model used in imputation (default is None for no depth limit).
