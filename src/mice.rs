use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::{HashMap, HashSet};
use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;


pub fn core_mice_impute(
    df: &DataFrame, 
    max_iterations: usize, 
    max_depth: Option<u16>
    ) -> PyResult<PyDataFrame> {

    println!(
        "Running Random Forest MICE with max_iterations={} and max_depth={:?}", 
        max_iterations, 
        max_depth
    );

    let n_rows = df.height();
    let n_cols = df.width();

    // Store indices of missing values per column index: col_idx -> Vec<row_idx>
    let mut missing_locations: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut imputed_columns: Vec<Column> = Vec::new();

    // 1. Identify missing locations and initialize with the column mean
    for (col_idx, series) in df.materialized_column_iter().enumerate() {
        // We cast to Float64 as we are building numerical models
        let s_f64 = series.cast(&DataType::Float64)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let f64_chunked = s_f64.f64()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if f64_chunked.null_count() > 0 {
            // Collect the indices of missing rows
            let missing_indices: Vec<usize> = f64_chunked
                .iter()
                .enumerate()
                .filter_map(|(i, val_opt)| if val_opt.is_none() { Some(i) } else { None })
                .collect();
            
            missing_locations.insert(col_idx, missing_indices);

            // Calculate mean and fill nulls for the initial state
            let mean_val = f64_chunked.mean().unwrap_or(0.0);
            let filled_ca: Float64Chunked = f64_chunked
                .iter()
                .map(|opt_v| Some(opt_v.unwrap_or(mean_val)))
                .collect();
            
            let mut filled_s = filled_ca.into_series();
            filled_s.rename(series.name().clone());
            
            imputed_columns.push(Column::from(filled_s));
        } else {
            imputed_columns.push(Column::from(s_f64));
        }
    }

    // Process the missing columns in a deterministic order (left-to-right)
    let mut missing_cols: Vec<usize> = missing_locations.keys().copied().collect();
    missing_cols.sort();

    // 2. Chained Equations Loop
    for _iteration in 0..max_iterations {
        // used for debugging and progress tracking
        // println!("  -> MICE Iteration {}/{}", _iteration + 1, max_iterations);

        for &col_idx in &missing_cols {
            let missing_rows = missing_locations.get(&col_idx).unwrap();
            let missing_set: HashSet<usize> = missing_rows.iter().copied().collect();

            // Cache the chunked arrays for fast row-level access during this step directly from current_columns
            let f64_columns: Vec<&Float64Chunked> = imputed_columns
                .iter()
                .map(|c| c.as_materialized_series().f64().unwrap())
                .collect();

            let mut x_train = Vec::new();
            let mut y_train = Vec::new();
            let mut x_test = Vec::new();

            // Build the training and testing datasets
            for row_idx in 0..n_rows {
                let mut row_features = Vec::with_capacity(n_cols - 1);
                
                for f_col_idx in 0..n_cols {
                    if f_col_idx == col_idx { continue; } // Skip the target column
                    // All columns are fully imputed now, so unwrap_or is a safety fallback
                    let val = f64_columns[f_col_idx].get(row_idx).unwrap_or(0.0);
                    row_features.push(val);
                }

                if missing_set.contains(&row_idx) {
                    x_test.push(row_features);
                } else {
                    x_train.push(row_features);
                    let target_val = f64_columns[col_idx].get(row_idx).unwrap_or(0.0);
                    y_train.push(target_val);
                }
            }

            // Convert raw Vecs into Smartcore dense matrices
            let x_train_mat = DenseMatrix::from_2d_vec(&x_train);
            let x_test_mat = DenseMatrix::from_2d_vec(&x_test);

            // Configure Tree parameters
            let mut params = RandomForestRegressorParameters::default();
            if let Some(depth) = max_depth {
                params = params.with_max_depth(depth);
            }

            // Train Random Forest and Predict
            let rf = RandomForestRegressor::fit(&x_train_mat, &y_train, params)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("RF Train Error: {}", e)))?;
            
            let preds = rf.predict(&x_test_mat)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("RF Predict Error: {}", e)))?;

            // Update the specific column with the newly predicted values
            let target_ca = f64_columns[col_idx];
            let mut target_vec: Vec<Option<f64>> = target_ca.iter().collect();
            
            for (i, &row_idx) in missing_rows.iter().enumerate() {
                target_vec[row_idx] = Some(preds[i]);
            }

            let new_ca: Float64Chunked = target_vec.into_iter().collect();
            let mut new_series = new_ca.into_series();
            new_series.rename(imputed_columns[col_idx].name().clone());

            // Rebuild the Column vector in-place
            imputed_columns[col_idx] = Column::from(new_series);
        }
    }

    // 3. Build the DataFrame once at the very end
    let final_df = DataFrame::new(n_rows, imputed_columns)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyDataFrame(final_df))
}
