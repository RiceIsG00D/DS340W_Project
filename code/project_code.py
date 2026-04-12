import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "./data"
RANDOM_STATE = 25


def load_all_csvs(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print("CSV count:", len(csv_files))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            df["source_file"] = os.path.basename(path)
            frames.append(df)
            print(f"Loaded: {os.path.basename(path)} | shape={df.shape}")
        except Exception as exc:
            print(f"Skipping {os.path.basename(path)} because of error: {exc}")

    if not frames:
        raise ValueError("CSV files were found, but none could be loaded.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged


def find_first_match(columns, candidates):
    lower_cols = {c.lower().strip(): c for c in columns}

    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_cols:
            return lower_cols[key]

    for cand in candidates:
        key = cand.lower().strip()
        for col in columns:
            if key in col.lower().strip():
                return col

    return None


def convert_comma_decimal(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace(["", "nan", "None", "NULL"], np.nan),
        errors="coerce",
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    colmap = {}

    athlete_col = find_first_match(out.columns, [
        "Athlete", "athlete", "Athlete Name", "athlete_name", "Name"
    ])
    date_col = find_first_match(out.columns, ["Date", "date"])
    activity_col = find_first_match(out.columns, [
        "Activity Type", "activity_type", "Activity", "sport"
    ])
    tss_col = find_first_match(out.columns, [
        "Training Stress Score®", "Training Stress Score", "TSS", "stress score"
    ])
    sleep_col = find_first_match(out.columns, [
        "Sleep Score", "sleep score", "Sleep", "sleep", "Sleep Quality"
    ])
    resting_hr_col = find_first_match(out.columns, [
        "Resting HR", "resting hr", "Rest HR", "rest_hr"
    ])

    if athlete_col:
        colmap[athlete_col] = "athlete"
    if date_col:
        colmap[date_col] = "date"
    if activity_col:
        colmap[activity_col] = "activity_type"
    if tss_col:
        colmap[tss_col] = "target_tss"
    if sleep_col:
        colmap[sleep_col] = "sleep_score"
    if resting_hr_col:
        colmap[resting_hr_col] = "resting_hr"

    out = out.rename(columns=colmap)

    if "athlete" not in out.columns:
        out["athlete"] = out["source_file"].str.replace(".csv", "", regex=False)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    skip_cols = {
        "athlete", "activity_type", "date", "source_file",
        "target_tss"
    }

    for col in out.columns:
        if col in skip_cols:
            continue
        if out[col].dtype == object:
            converted = convert_comma_decimal(out[col])
            non_null_original = out[col].notna().sum()
            non_null_converted = converted.notna().sum()
            if non_null_original > 0 and (non_null_converted / non_null_original) >= 0.6:
                out[col] = converted

    if "target_tss" in out.columns:
        out["target_tss"] = convert_comma_decimal(out["target_tss"])
    if "sleep_score" in out.columns:
        out["sleep_score"] = convert_comma_decimal(out["sleep_score"])
    if "resting_hr" in out.columns:
        out["resting_hr"] = convert_comma_decimal(out["resting_hr"])

    return out


def add_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sort_cols = [c for c in ["athlete", "activity_type", "date"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    group_cols = ["athlete"] if "athlete" in out.columns else []
    if "activity_type" in out.columns:
        group_cols.append("activity_type")

    if not group_cols:
        out["proxy_load_3"] = np.nan
        out["proxy_load_7"] = np.nan
        out["proxy_load_lag1"] = np.nan
        out["proxy_load_lag3"] = np.nan
        out["proxy_tss_std_7"] = np.nan
        out["proxy_rest_gap_days"] = np.nan
        out["proxy_sleep_3"] = np.nan
        out["proxy_sleep_lag1"] = np.nan
        out["proxy_fatigue_flag"] = np.nan
        return out

    grp = out.groupby(group_cols, dropna=False, group_keys=False)

    if "target_tss" in out.columns:
        out["proxy_load_3"] = grp["target_tss"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out["proxy_load_7"] = grp["target_tss"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out["proxy_load_lag1"] = grp["target_tss"].shift(1)
        out["proxy_load_lag3"] = grp["target_tss"].shift(3)
        out["proxy_tss_diff_1"] = grp["target_tss"].diff()
        out["proxy_tss_std_7"] = grp["target_tss"].transform(lambda s: s.rolling(7, min_periods=2).std())
        median_load = out["proxy_load_3"].median(skipna=True)
        out["proxy_fatigue_flag"] = (out["proxy_load_3"] > median_load).astype(int)
    else:
        out["proxy_load_3"] = np.nan
        out["proxy_load_7"] = np.nan
        out["proxy_load_lag1"] = np.nan
        out["proxy_load_lag3"] = np.nan
        out["proxy_tss_diff_1"] = np.nan
        out["proxy_tss_std_7"] = np.nan
        out["proxy_fatigue_flag"] = np.nan

    if "sleep_score" in out.columns:
        out["proxy_sleep_3"] = grp["sleep_score"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out["proxy_sleep_lag1"] = grp["sleep_score"].shift(1)
    else:
        out["proxy_sleep_3"] = np.nan
        out["proxy_sleep_lag1"] = np.nan

    if "date" in out.columns:
        out["proxy_rest_gap_days"] = grp["date"].diff().dt.days
    else:
        out["proxy_rest_gap_days"] = np.nan

    return out


def choose_feature_sets(df: pd.DataFrame):
    exclude_cols = {"target_tss", "date", "source_file"}

    numeric_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols and df[c].notna().sum() >= max(10, int(0.2 * len(df)))
    ]
    categorical_features = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in exclude_cols and df[c].notna().sum() >= max(10, int(0.2 * len(df)))
    ]

    baseline_numeric = [c for c in numeric_features if not c.startswith("proxy_") and c not in {"sleep_score", "resting_hr"}]
    baseline_categorical = categorical_features.copy()

    improved_numeric = numeric_features.copy()
    improved_categorical = categorical_features.copy()

    return baseline_numeric, baseline_categorical, improved_numeric, improved_categorical


def chronological_split(work: pd.DataFrame, test_size: float = 0.2):
    if "date" in work.columns and work["date"].notna().sum() >= 10:
        ordered = work.sort_values("date").reset_index(drop=True)
        split_idx = int(len(ordered) * (1 - test_size))
        train_df = ordered.iloc[:split_idx].copy()
        test_df = ordered.iloc[split_idx:].copy()
        return train_df, test_df

    train_df, test_df = train_test_split(work, test_size=test_size, random_state=RANDOM_STATE)
    return train_df.copy(), test_df.copy()


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )


def evaluate_model(
    df: pd.DataFrame,
    numeric_features,
    categorical_features,
    model_kind: str,
    target: str = "target_tss",
):
    features = numeric_features + categorical_features
    if not features:
        print(f"No usable features found for {model_kind}.")
        return None

    keep_cols = [target] + features
    if "date" in df.columns:
        keep_cols.append("date")

    work = df[keep_cols].copy()
    work = work[work[target].notna()].copy()

    if len(work) < 10:
        print(f"Not enough rows to run {model_kind}.")
        return None

    train_df, test_df = chronological_split(work)
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    if model_kind == "baseline_lasso":
        model = Lasso(alpha=0.04)
    elif model_kind == "enhanced_elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 0.95, 1.0],
            alphas=np.logspace(-3, 1, 40),
            cv=5,
            max_iter=20000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }
    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds,
    }).reset_index(drop=True)

    return metrics, pred_df, pipe


def plot_results(comparison: pd.DataFrame, baseline_preds: pd.DataFrame, improved_preds: pd.DataFrame):
    for metric in ["MAE", "RMSE", "R2"]:
        if metric in comparison.columns:
            plt.figure(figsize=(7, 4))
            comparison.set_index("Model")[metric].plot(kind="bar")
            plt.title(f"{metric} Comparison")
            plt.ylabel(metric)
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.show()

    plt.figure(figsize=(7, 5))
    if baseline_preds is not None:
        plt.scatter(baseline_preds["actual"], baseline_preds["predicted"], alpha=0.7, label="Baseline")
    plt.scatter(improved_preds["actual"], improved_preds["predicted"], alpha=0.7, label="Enhanced")

    all_series = [improved_preds["actual"], improved_preds["predicted"]]
    if baseline_preds is not None:
        all_series.extend([baseline_preds["actual"], baseline_preds["predicted"]])
    all_vals = pd.concat(all_series, ignore_index=True)
    mn, mx = all_vals.min(), all_vals.max()
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual TSS")
    plt.ylabel("Predicted TSS")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

    n = len(improved_preds)
    plot_df = pd.DataFrame({
        "Actual": improved_preds["actual"].iloc[:n].reset_index(drop=True),
        "Enhanced Predicted": improved_preds["predicted"].iloc[:n].reset_index(drop=True),
    })
    if baseline_preds is not None:
        n = min(len(baseline_preds), len(improved_preds))
        plot_df = pd.DataFrame({
            "Actual": baseline_preds["actual"].iloc[:n].reset_index(drop=True),
            "Baseline Predicted": baseline_preds["predicted"].iloc[:n].reset_index(drop=True),
            "Enhanced Predicted": improved_preds["predicted"].iloc[:n].reset_index(drop=True),
        })

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df.index, plot_df["Actual"], label="Actual")
    if "Baseline Predicted" in plot_df.columns:
        plt.plot(plot_df.index, plot_df["Baseline Predicted"], label="Baseline Predicted")
    plt.plot(plot_df.index, plot_df["Enhanced Predicted"], label="Enhanced Predicted")
    plt.xlabel("Test Observation")
    plt.ylabel("TSS")
    plt.title("Prediction Comparison on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return plot_df


def main():
    df_raw = load_all_csvs(DATA_DIR)
    print("\nRaw combined shape:", df_raw.shape)

    df = standardize_columns(df_raw)
    df = add_proxy_features(df)

    if "target_tss" not in df.columns:
        raise ValueError("Could not find a TSS / Training Stress Score column.")

    baseline_numeric, baseline_categorical, improved_numeric, improved_categorical = choose_feature_sets(df)

    print("\nBaseline numeric features:")
    print(baseline_numeric)
    print("\nBaseline categorical features:")
    print(baseline_categorical)
    print("\nEnhanced numeric features:")
    print(improved_numeric)
    print("\nEnhanced categorical features:")
    print(improved_categorical)

    baseline_result = evaluate_model(
        df,
        baseline_numeric,
        baseline_categorical,
        model_kind="baseline_lasso",
    )
    improved_result = evaluate_model(
        df,
        improved_numeric,
        improved_categorical,
        model_kind="enhanced_elasticnet",
    )

    if improved_result is None:
        raise ValueError("Enhanced model could not run.")

    rows = []
    baseline_preds = None
    if baseline_result is not None:
        baseline_metrics, baseline_preds, _ = baseline_result
        print("\nBaseline metrics:", baseline_metrics)
        rows.append({"Model": "Baseline Lasso", **baseline_metrics})

    improved_metrics, improved_preds, improved_model = improved_result
    print("\nEnhanced metrics:", improved_metrics)
    rows.append({"Model": "Enhanced ElasticNetCV", **improved_metrics})

    comparison = pd.DataFrame(rows)
    print("\nComparison table:")
    print(comparison)

    plot_df = plot_results(comparison, baseline_preds, improved_preds)

    comparison.to_csv("model_metric_comparison_enhanced.csv", index=False)
    if baseline_preds is not None:
        baseline_preds.to_csv("baseline_predictions.csv", index=False)
    improved_preds.to_csv("enhanced_predictions.csv", index=False)
    plot_df.to_csv("prediction_plot_values.csv", index=False)

    print("\nSaved output files:")
    print("- model_metric_comparison_enhanced.csv")
    if baseline_preds is not None:
        print("- baseline_predictions.csv")
    print("- enhanced_predictions.csv")
    print("- prediction_plot_values.csv")

    if hasattr(improved_model.named_steps["model"], "alpha_"):
        print("\nChosen ElasticNet alpha:", improved_model.named_steps["model"].alpha_)
        print("Chosen ElasticNet l1_ratio:", improved_model.named_steps["model"].l1_ratio_)


if __name__ == "__main__":
    main()
