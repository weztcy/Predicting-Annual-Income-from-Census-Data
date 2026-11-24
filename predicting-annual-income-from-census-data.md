# Portfolio Project Data — Predicting Annual Income from Census Data

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/predicting_annual_income_from/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image14.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "predicting-annual-income-from-census-data",
  "code": "PRO-23 // ML-08",
  "title": "Predicting Annual Income from Census Data",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Predicting-Annual-Income-from-Census-Data",
  "license": "MIT License",
  "banner": "/images/projects/predicting_annual_income_from/banner.webp",
  "summary": "Leakage-free supervised ML pipeline predicting >$50K census income with tuned HistGradientBoosting, permutation importance, and fairness audit.",
  "projectDescription": "End-to-end supervised classification system on 48,790 cleaned 1994 US Adult Census records predicting annual income (>50K vs. <=50K). Integrates ColumnTransformer preprocessing, 5-fold CV across 9 candidates, GridSearchCV tuning, fairness auditing, and a Streamlit UI.",
  "architectureSummary": "Adult.csv -> Cleaning & Deduplication -> Stratified 80:20 Split -> ColumnTransformer -> Tuned HistGradientBoosting Pipeline -> Evaluation, Fairness Audit & Streamlit UI",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Joblib",
    "Streamlit",
    "Pytest"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Classification Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "Tabular Socioeconomic Classification",
    "architecturePipeline": "ColumnTransformer + HistGradientBoosting",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Supervised Tabular Classification Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Socioeconomic Census Income Prediction",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "Binary Income Classification",
      "Permutation Feature Importance",
      "Demographic Fairness Audit"
    ]
  },
  "mlProblemContext": "Classifying annual income from 1994 US Census records poses significant tabular modeling challenges, including a 3.18:1 class imbalance, '?' missing placeholders, 1:1 feature redundancy between education and education-num, zero-inflated capital gains top-coded at $99,999, and historical demographic disparities.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Adult Census Dataset",
    "source": "UCI Machine Learning Repository (1994 US Census, Barry Becker & Ronny Kohavi)",
    "modality": "Tabular",
    "scale": "48,842 Raw Rows / 48,790 Cleaned Rows (39,032 Train / 9,758 Test)",
    "inputShape": "N x 12 Retained Features (5 Numeric, 7 Categorical)",
    "dataFormat": "CSV (Adult.csv) & JSON Metadata",
    "description": "Contains 48,790 deduplicated adult census records split 80:20 with target stratification (`<=50K`: 76.06%, `>50K`: 23.94%). Features span demographics, education, occupation, weekly hours, and capital gains/losses.",
    "provenanceNote": "UCI ML Repository Dataset ID 2 (DOI: 10.24432/C5XW20); restricted strictly to academic benchmarking and prohibited for real-world screening."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Mixed Tabular Feature Space",
    "spaceNote": "Drops redundant `education` string (1:1 with `education-num`) and census sampling weight `fnlwgt`.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "income",
      "type": "Binary Class (<=50K vs. >50K)",
      "distribution": "<=50K: 37,109 (76.06%) | >50K: 11,681 (23.94%)"
    },
    "inputSignals": [
      {
        "name": "capital-gain",
        "type": "Numerical",
        "importanceOrRole": "Rank #1 (0.2436 PR-AUC Drop)",
        "description": "Annual investment capital gains in USD; 91.7% zero-inflated and top-coded at $99,999."
      },
      {
        "name": "marital-status",
        "type": "Categorical",
        "importanceOrRole": "Rank #2 (0.1376 PR-AUC Drop)",
        "description": "Marital status categories; `Married-civ-spouse` strongly correlates with dual-income households."
      },
      {
        "name": "education-num",
        "type": "Numerical (Ordinal)",
        "importanceOrRole": "Rank #3 (0.0676 PR-AUC Drop)",
        "description": "Ordinal years/level of formal education from 1 (Preschool) to 16 (Doctorate); r = 0.333 with >50K."
      },
      {
        "name": "age",
        "type": "Numerical",
        "importanceOrRole": "Rank #4 (0.0631 PR-AUC Drop)",
        "description": "Individual age in years; median 43-44 years for >50K vs. 34 years for <=50K."
      },
      {
        "name": "capital-loss",
        "type": "Numerical",
        "importanceOrRole": "Rank #5 (0.0459 PR-AUC Drop)",
        "description": "Annual investment capital losses in USD; 95.3% zero-inflated across census records."
      },
      {
        "name": "occupation",
        "type": "Categorical",
        "importanceOrRole": "Rank #6 (0.0307 PR-AUC Drop)",
        "description": "Employment role (`Exec-managerial` 47.8% >50K vs. `Priv-house-serv` 1.2% >50K); 5.75% missing."
      },
      {
        "name": "hours-per-week",
        "type": "Numerical",
        "importanceOrRole": "Rank #7 (0.0202 PR-AUC Drop)",
        "description": "Weekly working hours (1-99); >50K earners show a heavy right tail between 45 and 60 hours."
      }
    ],
    "dataCharacteristicsSummary": "Across 48,790 deduplicated records, income (>50K) correlates positively with education-num (r = 0.333), age (r = 0.230), hours-per-week (r = 0.228), and capital-gain (r = 0.223), while fnlwgt exhibits near-zero correlation (r = -0.006). Capital-gain and capital-loss exhibit 91.7% and 95.3% zero inflation."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Binary Classification with Imbalance-Aware Ranking Optimization",
    "methodologyDescription": "Nine candidate configurations spanning Dummy, Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting classifiers (with default and balanced class weights) were benchmarked via 5-fold stratified CV on 39,032 training samples, selecting HistGradientBoosting for GridSearchCV PR-AUC tuning.",
    "justification": "Histogram-based gradient boosting natively captures non-linear interactions across zero-inflated financial features and dense one-hot socioeconomic categories, outperforming linear and single-tree models on both PR-AUC (0.8408 vs. 0.7683-0.8137) and ROC-AUC (0.9327)."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "String Trimming & Missing Standardization",
      "description": "Strips leading/trailing whitespace on string columns and converts `?`, `nan`, and `None` placeholders to `np.nan` without dropping partial rows.",
      "badge": "Cleaning"
    },
    {
      "step": "02",
      "title": "Target Normalization & Exact Deduplication",
      "description": "Strips trailing periods from target strings (`<=50K.`, `>50K.`) to enforce strict binary labels and removes 52 identical duplicate rows (`48,842` -> `48,790`).",
      "badge": "Deduplication"
    },
    {
      "step": "03",
      "title": "Redundant & Noise Column Removal",
      "description": "Drops `education` (1:1 collinear with ordinal `education-num`) and census sampling weight `fnlwgt` (`r = -0.006` linear correlation with target).",
      "badge": "Feature Pruning"
    },
    {
      "step": "04",
      "title": "Stratified Split & ColumnTransformer Pipeline",
      "description": "Executes an 80:20 stratified split (`random_state=42`) prior to fitting median imputation + StandardScaler (numeric) and mode imputation + OneHotEncoder (categorical).",
      "badge": "Zero Leakage"
    }
  ],
  "mlDataPreparationNote": "Stratified train-test splitting occurs strictly prior to imputation, scaling, and one-hot encoding inside scikit-learn Pipelines to prevent cross-split leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Best Tuned HistGradientBoostingClassifier Pipeline",
    "architectureSummary": "Scikit-learn Pipeline combining a ColumnTransformer (median imputer + StandardScaler for 5 numeric features; most-frequent imputer + OneHotEncoder for 7 categorical features) with a histogram-based gradient boosted decision tree ensemble.",
    "configurationBadge": "GridSearchCV Tuned",
    "hyperparameters": [
      {
        "name": "learning_rate",
        "value": "0.08",
        "description": "Shrinkage step size selected from [0.08, 0.1, 0.12] to balance convergence and generalization."
      },
      {
        "name": "max_iter",
        "value": "150",
        "description": "Maximum boosting iterations selected from [100, 150] for ensemble capacity."
      },
      {
        "name": "max_leaf_nodes",
        "value": "31",
        "description": "Maximum leaf nodes per tree selected from [31, 45] to control tree complexity."
      },
      {
        "name": "l2_regularization",
        "value": "1.0",
        "description": "L2 leaf penalty selected from [0.0, 1.0] to mitigate overfitting on one-hot categories."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixed deterministic seed across dataset splitting, CV folds, and model training."
      }
    ],
    "objectiveCriterion": "Log-Loss (Binary Cross-Entropy) with PR-AUC (Average Precision) GridSearchCV Scoring",
    "optimizationSolver": "Histogram-Based Gradient Boosting with Newton-Raphson Leaf Updates",
    "trainingProcess": "A 24-combination GridSearchCV with 5-fold StratifiedKFold (random_state=42) optimized Average Precision (PR-AUC) exclusively on training folds, improving CV PR-AUC from 0.8254 to 0.8258. The winning HistGradientBoosting pipeline was refitted on all 39,032 training rows and serialized via Joblib."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "5-Fold Stratified Cross-Validation & Single-Pass Held-Out Test Evaluation",
    "description": "Nine model configurations were benchmarked via out-of-fold 5-fold stratified CV on 39,032 training samples, followed by a single evaluation pass on 9,758 untouched test records alongside 10-repeat permutation importance and subgroup fairness auditing.",
    "metrics": [
      {
        "label": "TEST ACCURACY",
        "value": "87.83%",
        "benchmark": "Baseline: 76.06%"
      },
      {
        "label": "TEST ROC-AUC",
        "value": "0.9327",
        "benchmark": "Baseline: 0.5000"
      },
      {
        "label": "TEST PR-AUC",
        "value": "0.8408",
        "benchmark": "Baseline: 0.2394"
      },
      {
        "label": "POSITIVE F1 (>50K)",
        "value": "0.7293",
        "benchmark": "Target > 0.70"
      },
      {
        "label": "BALANCED ACCURACY",
        "value": "81.20%",
        "benchmark": "Baseline: 50.00%"
      },
      {
        "label": "MACRO F1-SCORE",
        "value": "0.8254",
        "benchmark": "Baseline: 0.4320"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Interactive Prediction Form",
      "badge": "Web UI",
      "imageUrl": "/images/projects/predicting_annual_income_from/image01.webp",
      "telemetryNote": "12 census feature inputs with built-in academic ethics warning banner"
    },
    {
      "title": "Streamlit Prediction Output & Probability Breakdown",
      "badge": "Inference UI",
      "imageUrl": "/images/projects/predicting_annual_income_from/image02.webp",
      "telemetryNote": "Live >50K prediction card displaying 70.36% positive probability score"
    },
    {
      "title": "Streamlit Model Performance & Comparison Tab",
      "badge": "Benchmark UI",
      "imageUrl": "/images/projects/predicting_annual_income_from/image03.webp",
      "telemetryNote": "Test-set metrics table (N = 9,758) ranking Best Tuned HistGB at 0.8408 PR-AUC"
    },
    {
      "title": "Streamlit Fairness & Demographic Disparity Tab",
      "badge": "Fairness UI",
      "imageUrl": "/images/projects/predicting_annual_income_from/image04.webp",
      "telemetryNote": "Subgroup disparity chart paired with proxy-variable fairness takeaways"
    },
    {
      "title": "Age & Weekly Working Hours Distributions",
      "badge": "EDA Distribution",
      "imageUrl": "/images/projects/predicting_annual_income_from/image05.webp",
      "telemetryNote": "Median age 43 yrs (>50K) vs. 34 yrs (<=50K) and 45-60 hrs/wk right tail"
    },
    {
      "title": "Top Permutation Feature Importances (Best Model)",
      "badge": "Explainability",
      "imageUrl": "/images/projects/predicting_annual_income_from/image06.webp",
      "telemetryNote": "capital-gain (0.2436) and marital-status (0.1376) lead 10-repeat PR-AUC drop"
    },
    {
      "title": "Income Class Proportions Across Education Levels",
      "badge": "EDA Stratification",
      "imageUrl": "/images/projects/predicting_annual_income_from/image07.webp",
      "telemetryNote": ">50K rate rises from 1.2% (Preschool) to 41.3% (Bachelors) and 74.0% (Prof-school)"
    },
    {
      "title": "Occupation Categories Sorted by High-Income Rate",
      "badge": "EDA Stratification",
      "imageUrl": "/images/projects/predicting_annual_income_from/image08.webp",
      "telemetryNote": "Exec-managerial (47.8%) and Prof-specialty (45.1%) lead occupational >50K share"
    },
    {
      "title": "Target Class Balance & Feature Missingness Audit",
      "badge": "Data Quality",
      "imageUrl": "/images/projects/predicting_annual_income_from/image09.webp",
      "telemetryNote": "76.06% <=50K vs. 23.94% >50K (N = 48,790); missingness in 3 categorical columns"
    },
    {
      "title": "Capital Gain & Capital Loss Zero-Inflation Plots",
      "badge": "EDA Histogram",
      "imageUrl": "/images/projects/predicting_annual_income_from/image10.webp",
      "telemetryNote": "91.7% zero capital-gain, 95.3% zero capital-loss, and $99,999 top-coding spike"
    },
    {
      "title": "Test-Set ROC & Precision-Recall Curves",
      "badge": "Evaluation Curves",
      "imageUrl": "/images/projects/predicting_annual_income_from/image11.webp",
      "telemetryNote": "Best Tuned HistGB achieves 0.933 ROC-AUC and 0.841 PR-AUC vs. 0.239 baseline"
    },
    {
      "title": "Subgroup Fairness Disparity Across Sex and Race",
      "badge": "Fairness Audit",
      "imageUrl": "/images/projects/predicting_annual_income_from/image12.webp",
      "telemetryNote": "Selection rate 7.88% (Female) vs. 27.83% (Male); Black recall 52.21% vs. 69.23% White"
    },
    {
      "title": "Multi-Metric Model Comparison Bar Chart",
      "badge": "Model Benchmark",
      "imageUrl": "/images/projects/predicting_annual_income_from/image13.webp",
      "telemetryNote": "Compares all 10 models on Accuracy, F1_Positive, Balanced Accuracy, ROC-AUC, PR-AUC"
    },
    {
      "title": "Numeric Correlation Heatmap & Test Confusion Matrix",
      "badge": "Correlation & Matrix",
      "imageUrl": "/images/projects/predicting_annual_income_from/image14.webp",
      "telemetryNote": "93.9% specificity (6,970 TN / 452 FP) and 68.5% recall (1,600 TP / 736 FN)"
    }
  ],
  "mlVisualizationTelemetryNote": "5-fold CV (0.8258 PR-AUC) aligns closely with held-out test performance (0.8408 PR-AUC, 0.9327 ROC-AUC), confirming generalization without overfitting.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Permutation importance on the test set reveals capital-gain (0.2436) and marital-status (0.1376) as primary predictors, while sex (0.0013) and race (0.0014) have minimal direct drop. Yet ablating sex and race leaves subgroup disparities nearly intact because relationship and occupation act as strong proxies.",
    "keyDrivers": [
      {
        "factor": "capital-gain & capital-loss",
        "impact": "Combined 0.2895 PR-AUC drop; non-zero capital gains strongly differentiate >50K earners."
      },
      {
        "factor": "marital-status & relationship",
        "impact": "0.1469 combined PR-AUC drop; Married-civ-spouse and Husband/Wife roles reflect household income and act as gender proxies."
      },
      {
        "factor": "education-num & age",
        "impact": "0.0676 and 0.0631 PR-AUC drops; captures monotonic income scaling with formal schooling and mid-career experience."
      }
    ],
    "limitations": [
      {
        "title": "1994 Temporal Shift & Nominal Threshold",
        "description": "Census records date back to 1994, making the $50,000 nominal income threshold unrepresentative of modern wage inflation and labor markets."
      },
      {
        "title": "Zero-Inflation & Top-Coding at $99,999",
        "description": "91.7% zero values in capital-gain and census top-coding at $99,999 compress upper-tail wealth resolution."
      },
      {
        "title": "Historical Demographic Bias & Proxy Leakage",
        "description": "Female selection rate (7.88%) vs. Male (27.83%) and Black recall (52.21%) persist even when `sex` and `race` are ablated due to correlated proxies."
      }
    ],
    "futureImprovements": [
      {
        "title": "Post-Processing Threshold & Fairness Calibration",
        "target": "Apply group-specific threshold optimization or exponentiated-gradient fairness constraints to reduce TPR and selection gaps."
      },
      {
        "title": "SHAP Local Attribution & Probability Calibration",
        "target": "Add TreeSHAP per-instance explanations and Platt/isotonic calibration checks to refine decision thresholds above 0.50."
      }
    ],
    "discoveryNote": "Fairness ablation demonstrates empirically that fairness-through-unawareness (dropping `sex` and `race`) fails when proxy variables remain."
  },
  "mlModelArtifacts": [
    {
      "name": "best_income_pipeline.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "best_model_metadata.json",
      "format": "JSON"
    },
    {
      "name": "model_comparison.csv",
      "format": "CSV"
    },
    {
      "name": "cv_model_comparison.csv",
      "format": "CSV"
    },
    {
      "name": "fairness_report.csv",
      "format": "CSV"
    },
    {
      "name": "feature_importance.csv",
      "format": "CSV"
    },
    {
      "name": "sample_predictions.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Binary Classification",
    "modelArchitecture": "Histogram Gradient Boosting",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Stratified 5-Fold Cross Validation"
  }
}
```
