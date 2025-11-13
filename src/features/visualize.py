"""
Main module for this repo
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from . import features


def eval_feature_ablation(
    df_source: Union[pd.DataFrame, str, Path],
    drop_features: List[str],
    x_columns: Optional[List[str]] = None,
    x_categorical: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare baseline vs. retrain-without-selected-features for each product.

    Args:
        df_source: DataFrame (left intact) or path to an Excel file.
        drop_features: Columns to exclude in the ablation model (compared to baseline with all X).
        x_columns: Feature columns to use for baseline. Defaults to features.X_ALL.
        x_categorical: Subset of x_columns that are categorical. Defaults to features.X_METADATA_CATEGORICAL.
        target_columns: Products/targets to evaluate. Defaults to features.PRODUCTS.
        n_splits: KFold splits.
        random_state: Seed for KFold shuffling.

    Returns:
        DataFrame with one row per product (target) and these columns:
          baseline_r2_mean, baseline_r2_std, baseline_rmse_mean, baseline_rmse_std,
          drop_r2_mean, drop_r2_std, drop_rmse_mean, drop_rmse_std,
          delta_drop_r2_mean, delta_drop_r2_std, delta_drop_rmse_mean, delta_drop_rmse_std
    """
    # ---- resolve inputs / defaults
    if isinstance(df_source, (str, Path)):
        df_raw = pd.read_excel(Path(df_source))
    else:
        df_raw = df_source.copy()

    df_raw = df_raw.dropna().copy()

    if x_columns is None:
        x_columns = list(features.X_ALL)
    if x_categorical is None:
        x_categorical = list(features.X_METADATA_CATEGORICAL)
    if target_columns is None:
        target_columns = list(features.PRODUCTS)

    present_x = [c for c in x_columns if c in df_raw.columns]
    if not present_x:
        raise ValueError("None of the requested x_columns are present in the dataframe.")
    X_full = df_raw[present_x].copy()

    cat_features_present = [c for c in x_categorical if c in X_full.columns]

    targets_to_use = [t for t in target_columns if t in df_raw.columns]
    if not targets_to_use:
        raise ValueError("None of the requested target_columns are present in the dataframe.")

    print(f"n_samples={len(X_full)}, n_features={X_full.shape[1]}")
    print("Targets to evaluate:", targets_to_use)

    # sanitize drop_features to existing columns
    drop_features_present = [c for c in drop_features if c in X_full.columns]
    missing = [c for c in drop_features if c not in X_full.columns]
    if missing:
        print(f"[WARN] The following drop_features are not present in X and will be ignored: {missing}")

    # ---- helpers (all internal)
    def _make_model() -> XGBRegressor:
        return XGBRegressor(enable_categorical=True, random_state=27)

    def _build_pipeline(cols: List[str], cat_present: List[str]) -> Pipeline:
        cat_in = [c for c in cols if c in cat_present]
        num_in = [c for c in cols if c not in cat_in]
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_in),
                ("num", "passthrough", num_in),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        return Pipeline([("pre", pre), ("model", _make_model())])

    def _get_cv_splits(n_samples: int, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        idx = np.arange(n_samples)
        return list(kf.split(idx))

    def _cv_scores(
        X: pd.DataFrame,
        y: np.ndarray,
        cols: List[str],
        cat_present: List[str],
        splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        r2_list, rmse_list = [], []
        for train_idx, val_idx in splits:
            X_tr = X.iloc[train_idx][cols]
            X_va = X.iloc[val_idx][cols]
            y_tr = y[train_idx]
            y_va = y[val_idx]

            pipe = _build_pipeline(cols, cat_present)
            pipe.fit(X_tr, y_tr)
            yhat = pipe.predict(X_va)

            r2_list.append(r2_score(y_va, yhat))
            rmse_list.append(root_mean_squared_error(y_va, yhat))
        return np.asarray(r2_list, float), np.asarray(rmse_list, float)

    def _mean_std(arr_like: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        arr = np.asarray(arr_like, dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return mu, sd

    # ---- main loop over products; one row per product
    rows: List[Dict[str, float]] = []
    for target in targets_to_use:
        print("\n" + "=" * 80)
        print(f"TARGET: {target}")
        print("=" * 80)

        y_full = df_raw[target].to_numpy()
        splits = _get_cv_splits(n_samples=len(X_full), n_folds=n_splits, seed=random_state)

        # Baseline: train with all x_columns present
        base_cols = X_full.columns.tolist()
        base_r2, base_rmse = _cv_scores(
            X=X_full, y=y_full, cols=base_cols, cat_present=cat_features_present, splits=splits
        )
        base_r2_m, base_r2_s = _mean_std(base_r2)
        base_rmse_m, base_rmse_s = _mean_std(base_rmse)

        print(f"\nBaseline (10-fold CV) for {target}:")
        print(f"  R²   = {base_r2_m:.4f} ± {base_r2_s:.4f}")
        print(f"  RMSE = {base_rmse_m:.6f} ± {base_rmse_s:.6f}")

        # Print the exact features requested to drop (after baseline)
        print("\nFeatures to drop (evaluated in the retrain-without-feature model):")
        print(drop_features_present)

        # Retrain without selected features
        cols_keep = [c for c in base_cols if c not in drop_features_present]
        if len(cols_keep) == 0:
            print("[WARN] Dropping all features leaves no predictors; results will be NaN.")
        drop_r2, drop_rmse = _cv_scores(
            X=X_full, y=y_full, cols=cols_keep, cat_present=cat_features_present, splits=splits
        )

        # Deltas per fold
        delta_r2 = drop_r2 - base_r2
        delta_rmse = drop_rmse - base_rmse

        row = {
            "product": target,
            "baseline_r2_mean": float(base_r2_m),
            "baseline_r2_std": float(base_r2_s),
            "baseline_rmse_mean": float(base_rmse_m),
            "baseline_rmse_std": float(base_rmse_s),
            "drop_r2_mean": float(drop_r2.mean()),
            "drop_r2_std": float(drop_r2.std(ddof=1)) if len(drop_r2) > 1 else 0.0,
            "drop_rmse_mean": float(drop_rmse.mean()),
            "drop_rmse_std": float(drop_rmse.std(ddof=1)) if len(drop_rmse) > 1 else 0.0,
            "delta_drop_r2_mean": float(delta_r2.mean()),
            "delta_drop_r2_std": float(delta_r2.std(ddof=1)) if len(delta_r2) > 1 else 0.0,
            "delta_drop_rmse_mean": float(delta_rmse.mean()),
            "delta_drop_rmse_std": float(delta_rmse.std(ddof=1)) if len(delta_rmse) > 1 else 0.0,
            "delta_r2_significant?": "✅" if float((-1) * delta_r2.mean()) > float(base_r2_s) else "❌",
            "delta_rmse_significant?": "✅" if float((-1) * delta_rmse.mean()) > float(base_rmse_s) else "❌",
        }
        rows.append(row)

    df_results = pd.DataFrame(rows).set_index("product")
    # Order columns for readability
    preferred = [
        "delta_drop_r2_mean",
        "delta_r2_significant?",
        "delta_rmse_significant?",
        "delta_drop_r2_std",
        "delta_drop_rmse_mean",
        "delta_drop_rmse_std",
        "drop_r2_mean",
        "drop_r2_std",
        "drop_rmse_mean",
        "drop_rmse_std",
        "baseline_r2_mean",
        "baseline_r2_std",
        "baseline_rmse_mean",
        "baseline_rmse_std",
    ]
    df_results = df_results.reindex(columns=preferred)
    try:
        from IPython.display import display  # type: ignore

        print("\n--- Retrain-without-selected-features (rows = product) ---")
        display(df_results)
    except Exception:
        pass

    return df_results


