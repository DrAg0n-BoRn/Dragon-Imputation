use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::{HashMap, HashSet};
use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use rayon::prelude::*;


fn build_matrices_parallel(
    f64_columns: &[&Float64Chunked],
    col_idx: usize,
    n_rows: usize,
    n_cols: usize,
    missing_set: &HashSet<usize>,
    missing_rows: &[usize],
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    // 1. Identify training rows
    let train_indices: Vec<usize> = (0..n_rows)
        .filter(|i| !missing_set.contains(i))
        .collect();

    // 2. Build training data concurrently
    let train_data: Vec<(Vec<f64>, f64)> = train_indices
        .par_iter()
        .map(|&row_idx| {
            let mut row_features = Vec::with_capacity(n_cols - 1);
            for f_col_idx in 0..n_cols {
                if f_col_idx == col_idx { continue; }
                row_features.push(f64_columns[f_col_idx].get(row_idx).unwrap_or(0.0));
            }
            let target_val = f64_columns[col_idx].get(row_idx).unwrap_or(0.0);
            (row_features, target_val)
        })
        .collect();

    let (x_train, y_train): (Vec<Vec<f64>>, Vec<f64>) = train_data.into_iter().unzip();

    // 3. Build testing data concurrently
    let x_test: Vec<Vec<f64>> = missing_rows
        .par_iter()
        .map(|&row_idx| {
            let mut row_features = Vec::with_capacity(n_cols - 1);
            for f_col_idx in 0..n_cols {
                if f_col_idx == col_idx { continue; }
                row_features.push(f64_columns[f_col_idx].get(row_idx).unwrap_or(0.0));
            }
            row_features
        })
        .collect();

    (x_train, y_train, x_test)
}


pub fn core_mice_impute(
    df: &DataFrame, 
    cat_cols: Vec<usize>,
    max_iterations: usize, 
    max_depth: Option<u16>,
    verbose: usize,
    seed: u64,
    n_jobs: Option<usize>
    ) -> PyResult<PyDataFrame> {

    let depth_display = max_depth.map(|d| d.to_string()).unwrap_or_else(|| "♾️ ".to_string());
    let jobs_display = n_jobs.map(|j| j.to_string()).unwrap_or_else(|| "All".to_string());
    let cat_display = if cat_cols.is_empty() {
        "None".to_string()
    } else {
        format!("{:?}", cat_cols)
    };


    if verbose >= 2 {
        println!("🐲 Running Random Forest MICE imputation with: \
            \n\t- Categorical columns={}, \
            \n\t- Max iterations={}, \
            \n\t- Max depth={}, \
            \n\t- Seed={}, \
            \n\t- CPU cores={}",
            cat_display,
            max_iterations, 
            depth_display,
            seed,
            jobs_display
        );
    }

    // Initialize custom thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs.unwrap_or(0)) 
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Thread pool error: {}", e)))?;

    let n_rows = df.height();
    let n_cols = df.width();

    // Create a HashSet for quick lookup of categorical column indices
    let cat_set: HashSet<usize> = cat_cols.into_iter().collect();

    // Store indices of missing values per column index: col_idx -> Vec<row_idx>
    let mut missing_locations: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut imputed_columns: Vec<Column> = Vec::new();

    // 1. Identify missing locations and initialize with the column mean
    for (col_idx, series) in df.materialized_column_iter().enumerate() {
        // Cast to Float64 to use numerical models
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

            // Calculate fill value (mode for categorical, mean for continuous)
            let is_categorical = cat_set.contains(&col_idx);
            let fill_val = if is_categorical {
                let mut counts: HashMap<u64, usize> = HashMap::new();
                for val_opt in f64_chunked.iter() {
                    if let Some(v) = val_opt {
                        *counts.entry(v.to_bits()).or_insert(0) += 1;
                    }
                }
                counts.into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(bits, _)| f64::from_bits(bits))
                    .unwrap_or(0.0)
            } else {
                f64_chunked.mean().unwrap_or(0.0)
            };
            
            let filled_ca: Float64Chunked = f64_chunked
                .iter()
                .map(|opt_v| Some(opt_v.unwrap_or(fill_val)))
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

    // Run the computationally heavy loop inside the thread pool
    pool.install(|| -> PyResult<()> {
        if verbose >= 3 {
            println!("\n🐲 Starting MICE iterations...");
        }
        // 2. Chained Equations Loop
        for iteration in 0..max_iterations {

            if verbose >= 3 {
                println!("  -> MICE Iteration {}/{}", iteration + 1, max_iterations);
            }

            for &col_idx in &missing_cols {
                let missing_rows = missing_locations.get(&col_idx).unwrap();
                let missing_set: HashSet<usize> = missing_rows.iter().copied().collect();

                // Cache the chunked arrays for fast row-level access during this step directly from current_columns
                let f64_columns: Vec<&Float64Chunked> = imputed_columns
                    .iter()
                    .map(|c| c.as_materialized_series().f64().unwrap())
                    .collect();

                // Delegate to the parallel helper to build the datasets
                let (x_train, y_train, x_test) = build_matrices_parallel(
                    &f64_columns,
                    col_idx,
                    n_rows,
                    n_cols,
                    &missing_set,
                    missing_rows,
                );

                // Convert raw Vecs into Smartcore dense matrices
                let x_train_mat = DenseMatrix::from_2d_vec(&x_train);
                let x_test_mat = DenseMatrix::from_2d_vec(&x_test);

                // Determine if the current column is categorical or continuous
                let is_categorical = cat_set.contains(&col_idx);

                let preds = if is_categorical {
                    // Train Random Forest Classifier and Predict
                    let mut params = RandomForestClassifierParameters::default();
                    // Offset seed by iteration and column to ensure trees vary across steps while remaining deterministic
                    params = params.with_seed(seed + (iteration as u64 * 1000) + col_idx as u64);

                    if let Some(depth) = max_depth {
                        params = params.with_max_depth(depth);
                    }

                    // Cast y_train to integers (u32) for the classifier since f64 lacks Ord trait
                    let y_train_cat: Vec<u32> = y_train.iter().map(|&y| y as u32).collect();

                    // Train Random Forest Classifier and Predict
                    let rf = RandomForestClassifier::fit(&x_train_mat, &y_train_cat, params)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("RF Classifier Train Error: {}", e)))?;
                    
                    let cat_preds: Vec<u32> = rf.predict(&x_test_mat)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("RF Classifier Predict Error: {}", e)))?;
                    
                    // Cast back to f64 so both arms of the if-else match types
                    cat_preds.into_iter().map(|p| p as f64).collect()
                } else {
                    // Train Random Forest Regressor and Predict
                    let mut params = RandomForestRegressorParameters::default();
                    // Offset seed by iteration and column to ensure trees vary across steps while remaining deterministic
                    params = params.with_seed(seed + (iteration as u64 * 1000) + col_idx as u64);

                    if let Some(depth) = max_depth {
                        params = params.with_max_depth(depth);
                    }

                    // Train Random Forest Regressor and Predict
                    let rf = RandomForestRegressor::fit(&x_train_mat, &y_train, params)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("RF Regressor Train Error: {}", e)))?;
                    
                    rf.predict(&x_test_mat)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("RF Regressor Predict Error: {}", e)))?
                };

                // Update the specific column with the newly predicted values and track convergence
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
        Ok(())
    })?;

    // 3. Build the DataFrame once at the very end
    let mut final_columns = Vec::with_capacity(n_cols);
    for (col_idx, col) in imputed_columns.into_iter().enumerate() {
        if cat_set.contains(&col_idx) {
            let casted_col = col.cast(&DataType::UInt32)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            final_columns.push(casted_col);
        } else {
            final_columns.push(col);
        }
    }

    let final_df = DataFrame::new(n_rows, final_columns)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    //report success
    if verbose >= 2 {
        println!("\n🐲 MICE imputation completed successfully.");
    }

    Ok(PyDataFrame(final_df))
}
