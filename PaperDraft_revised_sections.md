# PaperDraft Revised Sections

This file captures the paper updates that correspond to the code changes in `code/project_code.py`. Because the provided manuscript was a PDF rather than an editable source document, these sections are written as ready-to-paste replacement text for the report.

## 1. Assignment Update and Motivation

In the earlier version of our pipeline, the baseline model focused primarily on same-session training load variables. That provided a useful reference point, but it did not capture how performance is shaped by accumulated fatigue and recovery across multiple sessions. To better reflect the real dynamics of athletic performance, we updated the pipeline so that recovery-related features are computed at the athlete level and carried across time instead of being treated as isolated observations.

The revised implementation now includes rolling workload averages, lagged workload terms, short-term workload change, recent workload variability, rest-gap timing, and a new seven-day session-density feature. These additions were motivated by the idea that athlete performance depends not only on how hard an athlete trains, but also on how closely those sessions are clustered and how much time is available for recovery. This makes the feature set better aligned with the sports-science literature discussed in the paper.

## 2. Modular Block Replaced

For this assignment, we kept the baseline Lasso model as a reference model but strengthened the enhanced modeling block in two important ways. First, we continued using ElasticNetCV because it remains appropriate for correlated workload and recovery variables. Second, we updated the validation logic so that the enhanced model uses time-aware cross-validation when date information is available. This change better reflects the real prediction setting in which past sessions are used to estimate future performance.

We also revised the feature-selection logic to exclude high-cardinality text fields, such as workout titles, from categorical encoding. Those fields behave more like identifiers than generalizable predictors, and they can unnecessarily expand the design matrix while increasing the risk of overfitting. Removing them improves both computational efficiency and the interpretability of the resulting model.

## 3. Updated Methodology

The updated pipeline begins by loading and merging all athlete CSV files, standardizing shared fields such as athlete name, activity type, date, and Training Stress Score. After cleaning the data, the code generates both baseline and enhanced feature sets. The enhanced feature set includes athlete-level rolling averages over recent sessions, lagged training-load values, short-term workload change, rolling workload variability, days between sessions, and seven-day session density. Together, these features act as proxies for fatigue accumulation and recovery state.

In addition to feature engineering, the updated pipeline improves reproducibility and evaluation. The script now resolves file paths relative to the repository root, writes outputs into a dedicated `outputs` folder, and saves plots to files instead of requiring interactive windows. It also exports the learned coefficients from the enhanced model so the most influential features can be inspected directly. This is important because it preserves interpretability while still allowing the model to benefit from richer recovery-oriented predictors.

## 4. Discussion Addendum

The code changes strengthen the connection between the implementation and the claims made in the paper. Athlete-level recovery features are more realistic than activity-isolated proxies because fatigue and adaptation accumulate across the full training schedule rather than within a single activity label. The added seven-day session-density feature also provides a clearer representation of how compressed training can affect readiness and performance.

The runtime improvements are also methodologically meaningful. Excluding identifier-like text fields from the model reduces the chance that the algorithm memorizes workout-specific labels instead of learning stable performance relationships. In this sense, the code revisions do not only make the script faster; they also make the model more statistically defensible and easier to explain in the report.

## 5. Updated Results Sentence

When we reran the revised pipeline, the enhanced model continued to outperform the baseline model by a wide margin. In the verified local run, the baseline model produced `MAE = 18.0447`, `RMSE = 28.3261`, and `R2 = 0.2903`, while the enhanced ElasticNetCV model produced `MAE = 0.1483`, `RMSE = 0.5562`, and `R2 = 0.9997`. These results reinforce the conclusion that recovery-aware features substantially improve predictive performance.
