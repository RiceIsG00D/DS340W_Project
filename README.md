# Improving Athletic Performance Modeling Through Training, Recovery, and Sleep Proxies

This repository contains the code and athlete session data used to compare a baseline training-load model against an enhanced pipeline that adds recovery-oriented proxy features. The goal is to predict `Training Stress Score (TSS)` more realistically by accounting for the delayed effects of fatigue, recovery, and session spacing instead of relying on same-session workload alone.

## Project Summary

- Baseline model: `Lasso` using traditional training-related features.
- Enhanced model: `ElasticNetCV` using athlete-level rolling workload, lagged workload, variability, recovery-gap, and session-density features.
- Validation: chronological train/test split, with time-aware cross-validation for the enhanced model when date information is available.
- Outputs: metrics, prediction tables, coefficient export, and optional saved plots.

## Verified Results

The pipeline was run locally on April 19, 2026 with `python code/project_code.py --plot-mode skip`.

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Baseline Lasso | 18.0447 | 28.3261 | 0.2903 |
| Enhanced ElasticNetCV | 0.1483 | 0.5562 | 0.9997 |

These results support the paper's main claim that adding recovery-aware features substantially improves predictive performance over a training-only baseline.

## What Changed In The Code

The current version of the pipeline includes a few important improvements beyond the original script:

- Recovery features are computed at the athlete level, which better reflects how fatigue and recovery accumulate across sessions.
- A `proxy_session_density_7d` feature was added to capture how concentrated training has been over the previous week.
- The enhanced model now uses time-aware cross-validation instead of only a generic fold split.
- High-cardinality text fields such as workout titles are excluded from categorical modeling to reduce overfitting and dramatically improve runtime.
- Plots now default to file-based output instead of blocking on `plt.show()`, which makes the script run reliably in headless environments.
- The script exports model coefficients so the enhanced model remains interpretable for the report.

## Repository Layout

```text
DS340W_Project/
├── code/
│   └── project_code.py
├── data/
│   └── *.csv
├── outputs/                      # generated after running the script
├── PaperDraft_revised_sections.md
├── README.md
└── requirements.txt
```

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run The Pipeline

Default run with saved outputs:

```powershell
python code\project_code.py
```

Skip plots entirely:

```powershell
python code\project_code.py --plot-mode skip
```

Show plots interactively:

```powershell
python code\project_code.py --plot-mode show
```

## Generated Outputs

After a successful run, the script writes these files into `outputs/`:

- `model_metric_comparison_enhanced.csv`
- `baseline_predictions.csv`
- `enhanced_predictions.csv`
- `prediction_plot_values.csv`
- `enhanced_model_coefficients.csv`
- Plot images when `--plot-mode save` is used

## Paper Support

`PaperDraft_revised_sections.md` contains polished replacement text for the assignment-update sections of the paper. It explains why the code was changed, how the model block was updated, and how the revised methodology aligns with the paper's claims.
