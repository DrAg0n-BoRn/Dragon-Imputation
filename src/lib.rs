use pyo3::prelude::*;

pub mod mice;


/// A Python module implemented in Rust.
#[pymodule]
mod dragon_imputation {
    use pyo3::prelude::*;
    use polars::prelude::*;
    use pyo3_polars::PyDataFrame;
    use crate::mice;

    
    /// A Python function that performs MICE imputation on a Polars DataFrame with continuous and categorical variables in numerical form.
    /// 
    /// Arguments:
    ///     - df (PyDataFrame): The input Polars DataFrame with missing values.
    ///     - cat_cols (Sequence[int], optional): A sequence of column indices that are categorical. If None, all columns are treated as continuous. Defaults to None.
    ///     - max_iterations (int, optional): The maximum number of iterations for the MICE algorithm. Defaults to 30.
    ///     - max_depth (int, optional): The maximum depth of the trees in the Random Forest model. If None, no limit is set. Defaults to None.
    ///     - verbose (int, optional): The verbosity level for logging. Defaults to 2 standard logging.
    ///     - seed (int, optional): The random seed for reproducibility. Defaults to 42.
    ///     - n_jobs (int, optional): The number of threads to use for parallel processing. If None, it will use the number of available CPU cores.
    /// 
    /// Returns:
    ///     - PyDataFrame: A new Polars DataFrame with missing values imputed.
    #[pyfunction]
    #[pyo3(signature = (df, cat_cols=None, max_iterations=30, max_depth=None, verbose=2, seed=42, n_jobs=None))]
    pub fn mice_impute(
        df: PyDataFrame, 
        cat_cols: Option<Vec<usize>>,
        max_iterations: usize, 
        max_depth: Option<u16>,
        verbose: usize,
        seed: u64,
        n_jobs: Option<usize>
        ) -> PyResult<PyDataFrame> {
        // Convert to a Rust DataFrame at the boundary
        let r_df: DataFrame = df.into();

        let categorical_indices = cat_cols.unwrap_or_default();

        // delegate to the core MICE imputation function
        mice::core_mice_impute(&r_df, categorical_indices, max_iterations, max_depth, verbose, seed, n_jobs)
    }
}
