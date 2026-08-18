use pyo3::prelude::*;

pub mod mice;


/// A Python module implemented in Rust.
#[pymodule]
mod dragon_imputation {
    use pyo3::prelude::*;
    use polars::prelude::*;
    use pyo3_polars::PyDataFrame;
    use crate::mice;

    
    /// A Python function that performs MICE imputation on a Polars DataFrame.
    /// 
    /// Arguments:
    ///     - df (PyDataFrame): The input Polars DataFrame with missing values.
    ///     - max_iterations (int, optional): The maximum number of iterations for the MICE algorithm. Defaults to 10.
    ///     - max_depth (int, optional): The maximum depth of the trees in the Random Forest model. If None, no limit is set. Defaults to None.
    /// 
    /// Returns:
    ///     - PyDataFrame: A new Polars DataFrame with missing values imputed.
    #[pyfunction]
    #[pyo3(signature = (df, max_iterations=10, max_depth=None))]
    pub fn mice_impute(
        df: PyDataFrame, 
        max_iterations: usize, 
        max_depth: Option<u16>
        ) -> PyResult<PyDataFrame> {
        // Convert to a Rust DataFrame at the boundary
        let r_df: DataFrame = df.into();

        // delegate to the core MICE imputation function
        mice::core_mice_impute(&r_df, max_iterations, max_depth)
    }
}
