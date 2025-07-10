# MercadoLibre Challenge - New vs Used Classification

Machine learning solution to classify MercadoLibre items as "new" or "used".

## Results
- **Accuracy**: 87.3% (requirement: ≥86%)
- **Recall**: 95.0% (achieves 95% target!)
- **ROC-AUC**: 0.952

## Project Structure
```
├── data/
│   └── MLA_100k_checked_v3.jsonlines
├── docs/
│   ├── Opportunities_MeLi - CodeExercise DS_ML.docx
│   └── PRODUCTION_SUMMARY.md
├── notebooks/
│   ├── 01_eda_feature_selection.ipynb
│   └── 02_feature_engineering_pipeline.ipynb
├── src/
│   ├── new_or_used.py
│   ├── feature_engineering_pipeline.py
│   ├── model_experiments.py
│   └── xgboost_optimization.py
├── models/
│   ├── xgboost_optimized.json
│   └── feature_engineering_pipeline.pkl
├── results/
│   └── model_comparison_results.csv
└── requirements.txt
```

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
python src/feature_engineering_pipeline.py
python src/model_experiments.py
python src/xgboost_optimization.py
```

## What it does
- Loads and preprocesses the dataset
- Engineers 44 features from the raw data
- Compares 5 different models
- Optimizes XGBoost hyperparameters
- Achieves 87.3% accuracy and 95.0% recall (exceeds requirements)

## 🔬 Technical Details

### Feature Engineering (44 features)
- **Temporal Features**: Start time decomposition (hour, day of week, month)
- **Numerical Features**: Log-transformed price, inventory flags
- **Categorical Features**: One-hot encoding for buying mode, seller address, shipping mode
- **Text Features**: Warranty duration parsing, attribute counts
- **Seller Features**: Volume-based seller profiling
- **Business Features**: Payment method analysis, picture counts

### Model Architecture
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Optimization**: RandomizedSearchCV hyperparameter tuning + threshold optimization
- **Validation**: Stratified train-test split with proper preprocessing
- **Features**: 44 engineered features from comprehensive pipeline
- **Threshold**: Optimized to 0.3196 for 95% recall target

### Performance Metrics
| Metric | Value | Status |
|--------|-------|--------|
| Accuracy | 87.3% | ✅ Exceeds requirement (86%) |
| Recall | 95.0% | ✅ Achieves 95% target |
| Precision | 83.7% | ✅ Strong precision |
| ROC-AUC | 0.952 | ✅ Excellent discriminative power |
| F1-Score | 0.890 | ✅ Balanced performance |

## 📊 Notebooks

### notebooks/01_eda_feature_selection.ipynb
- Exploratory Data Analysis
- Feature importance analysis
- Data quality assessment
- Initial feature selection

### notebooks/02_feature_engineering_pipeline.ipynb
- Feature engineering development
- Pipeline construction
- Feature validation
- Preprocessing optimization

## 🏭 Production Usage

The `src/new_or_used.py` script contains the complete production pipeline:
- Loads trained models and feature engineering pipeline
- Processes new data through the same preprocessing steps
- Provides predictions with the optimized model

```bash
python src/new_or_used.py
```

## 🧪 Development

### Model Retraining
```bash
# Full pipeline retraining
python src/feature_engineering_pipeline.py
python src/model_experiments.py
python src/xgboost_optimization.py
```

## 📈 Key Achievements

- **Exceeds Requirements**: 87.3% accuracy exceeds the 86% requirement
- **Achieves Target**: 95.0% recall meets the proposed 95% business target
- **Robust Pipeline**: 44 engineered features with comprehensive preprocessing
- **Optimized Performance**: Hyperparameter tuning with threshold optimization
- **Production Ready**: Complete workflow from data loading to model deployment

---

**Last Updated**: 07-09-2025
**Model Version**: 1.0
**Performance**: 87.3% accuracy, 95.0% recall, 0.952 ROC-AUC
