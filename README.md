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
|-- code/
|   `-- project_code.py
|-- data/
|   `-- *.csv
|-- outputs/                      # generated after running the script
|-- README.md
`-- requirements.txt
```

## Run On A Fresh Computer

These steps are written for a clean computer that does not already have Python packages, a virtual environment, or a local copy of the project.

### 1. Install the required software

- Install Python 3.9 or newer from [python.org](https://www.python.org/downloads/).
- Install Git from [git-scm.com](https://git-scm.com/downloads).
- On Windows, make sure the Python installer adds Python to `PATH`.

Check that both are available:

```powershell
python --version
git --version
```

### 2. Download the repository

Clone the project from GitHub and move into the project folder:

```powershell
git clone https://github.com/RiceIsG00D/DS340W_Project.git
cd DS340W_Project
```

If you downloaded the repository as a ZIP instead of using Git, extract it and open a terminal inside the extracted `DS340W_Project` folder before continuing.

### 3. Create a virtual environment

This isolates the project dependencies from the rest of your machine.

On Windows:

```powershell
py -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the project dependencies

With the virtual environment active, install the required packages:

```powershell
pip install -r requirements.txt
```

### 5. Run the pipeline

The standard run command is:

```powershell
py code/project_code.py
```

This will:

- load the athlete CSV files from `data/`
- train the baseline and enhanced models
- print the evaluation metrics in the terminal
- save CSV outputs and plots into `outputs/`

### 6. Optional run modes

Skip plots entirely:

```powershell
py code/project_code.py --plot-mode skip
```

Show plots interactively instead of saving them:

```powershell
py code/project_code.py --plot-mode show
```

### 7. Confirm the run worked

After a successful run, you should see:

- model metrics printed in the terminal
- a new `outputs/` folder
- CSV files such as `model_metric_comparison_enhanced.csv`
- plot images if you used the default save mode

## Troubleshooting

- If `python` is not recognized, reinstall Python and enable the option to add it to `PATH`.
- If `pip install -r requirements.txt` fails, confirm that the virtual environment is activated.
- If you are on macOS or Linux and `python` is unavailable, use `python3` instead.
- If you want the simplest non-graphical run, use `python code/project_code.py --plot-mode skip`.

## Generated Outputs

After a successful run, the script writes these files into `outputs/`:

- `model_metric_comparison_enhanced.csv`
- `baseline_predictions.csv`
- `enhanced_predictions.csv`
- `prediction_plot_values.csv`
- `enhanced_model_coefficients.csv`
- plot images when the default save mode is used
