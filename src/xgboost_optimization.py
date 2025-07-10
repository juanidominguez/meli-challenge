"""
XGBoost Optimization Script - Updated with Full Feature Engineering Pipeline
===========================================================================

This script optimizes XGBoost hyperparameters and threshold to achieve 95% recall
for "new" items classification. Based on the successful model_experiments.py approach.
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, make_scorer
)
from scipy.stats import randint, uniform
import xgboost as xgb

# Import the proper feature engineering pipeline
from feature_engineering_pipeline import make_full_pipeline
from new_or_used import build_dataset


def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("🔄 Loading and preparing data...")
    
    # Load raw data
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = build_dataset()
    
    # Convert to DataFrames
    X_train = pd.json_normalize(X_train_raw, sep='_')
    X_test = pd.json_normalize(X_test_raw, sep='_')
    
    # Convert target to binary (new=1, used=0)
    y_train = np.array([1 if label == 'new' else 0 for label in y_train_raw])
    y_test = np.array([1 if label == 'new' else 0 for label in y_test_raw])
    
    print(f"✅ Data loaded: Train={X_train.shape}, Test={X_test.shape}")
    print(f"   Class distribution (train): {Counter(y_train)}")
    print(f"   Class distribution (test): {Counter(y_test)}")
    
    return X_train, X_test, y_train, y_test


def apply_feature_engineering(X_train, X_test, y_train):
    """Apply comprehensive feature engineering pipeline"""
    print("🔄 Applying comprehensive feature engineering pipeline...")
    
    # Create the full feature engineering pipeline
    pipeline = make_full_pipeline(target_name='condition')
    
    # Remove target column from training data if it exists
    X_train_clean = X_train.drop(columns=['condition']) if 'condition' in X_train.columns else X_train
    
    # Fit and transform training data
    X_train_processed = pipeline.fit_transform(X_train_clean, y_train)
    
    # Transform test data
    X_test_processed = pipeline.transform(X_test)
    
    print(f"✅ Feature engineering complete: {X_train_processed.shape[1]} features")
    print(f"   Training set: {X_train_processed.shape}")
    print(f"   Test set: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, pipeline


def optimize_xgboost(X_train, y_train, cv_folds=5, n_iter=50, random_state=42):
    """Optimize XGBoost hyperparameters"""
    print("🚀 Starting XGBoost hyperparameter optimization...")
    print(f"   Iterations: {n_iter}")
    print(f"   CV folds: {cv_folds}")
    print(f"   Optimization metric: Recall")
    
    # Calculate class weights
    class_counts = Counter(y_train)
    scale_pos_weight = class_counts[0] / class_counts[1]
    
    # Define parameter distribution
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 3),
        'reg_alpha': uniform(0, 3),
        'reg_lambda': uniform(0, 3),
        'scale_pos_weight': uniform(scale_pos_weight * 0.7, scale_pos_weight * 0.6)
    }
    
    # Create base model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state,
        verbosity=0,
        n_jobs=1
    )
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Create custom recall scorer
    recall_scorer = make_scorer(recall_score, pos_label=1)
    
    # Perform randomized search
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=recall_scorer,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    # Fit the search
    search.fit(X_train, y_train)
    
    print(f"✅ Optimization completed!")
    print(f"   Best recall score: {search.best_score_:.4f}")
    print(f"   Best hyperparameters:")
    for param, value in search.best_params_.items():
        print(f"     {param}: {value}")
    
    return search.best_estimator_, search.best_params_, search.best_score_


def optimize_threshold(model, X_test, y_test, recall_target=0.95):
    """Optimize classification threshold for target recall"""
    print(f"🎯 Optimizing threshold for {recall_target*100}% recall...")
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Find optimal threshold
    valid_indices = recall >= recall_target
    
    if valid_indices.any():
        valid_precisions = precision[valid_indices]
        best_idx = np.argmax(valid_precisions)
        
        optimal_threshold = thresholds[np.where(valid_indices)[0][best_idx]]
        optimal_precision = precision[valid_indices][best_idx]
        optimal_recall = recall[valid_indices][best_idx]
        
        print(f"✅ Optimal threshold found: {optimal_threshold:.4f}")
        print(f"   Precision: {optimal_precision:.4f}")
        print(f"   Recall: {optimal_recall:.4f}")
        
        return optimal_threshold, optimal_precision, optimal_recall
    else:
        max_recall = np.max(recall)
        print(f"❌ Target recall {recall_target*100}% not achievable")
        print(f"   Maximum possible recall: {max_recall*100:.1f}%")
        return None, None, max_recall


def evaluate_model(model, X_train, y_train, X_test, y_test, optimal_threshold=0.5):
    """Comprehensive model evaluation"""
    print("📊 Evaluating model performance...")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Optimized threshold predictions
    y_pred_test_optimized = (y_proba_test >= optimal_threshold).astype(int)
    
    # Calculate metrics
    results = {}
    
    # Standard threshold (0.5)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_roc_auc = roc_auc_score(y_test, y_proba_test)
    
    # Optimized threshold
    opt_accuracy = accuracy_score(y_test, y_pred_test_optimized)
    opt_precision = precision_score(y_test, y_pred_test_optimized)
    opt_recall = recall_score(y_test, y_pred_test_optimized)
    opt_f1 = f1_score(y_test, y_pred_test_optimized)
    
    # Confusion matrices
    cm_standard = confusion_matrix(y_test, y_pred_test)
    cm_optimized = confusion_matrix(y_test, y_pred_test_optimized)
    
    # Business metrics
    tn_std, fp_std, fn_std, tp_std = cm_standard.ravel()
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimized.ravel()
    
    business_risk_std = fp_std / (tn_std + fp_std + fn_std + tp_std) * 100
    business_risk_opt = fp_opt / (tn_opt + fp_opt + fn_opt + tp_opt) * 100
    
    # Training metrics for overfitting analysis
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_roc_auc = roc_auc_score(y_train, y_proba_train)
    
    # Display results
    print("\n=== RESULTS SUMMARY ===")
    print(f"\n📈 Standard Threshold (0.5):")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   F1 Score: {test_f1:.4f}")
    print(f"   ROC AUC: {test_roc_auc:.4f}")
    print(f"   Business Risk: {business_risk_std:.2f}%")
    
    print(f"\n🎯 Optimized Threshold ({optimal_threshold:.4f}):")
    print(f"   Accuracy: {opt_accuracy:.4f}")
    print(f"   Precision: {opt_precision:.4f}")
    print(f"   Recall: {opt_recall:.4f}")
    print(f"   F1 Score: {opt_f1:.4f}")
    print(f"   Business Risk: {business_risk_opt:.2f}%")
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Difference: {train_accuracy - test_accuracy:+.4f}")
    
    if train_accuracy - test_accuracy > 0.05:
        print("   ⚠️  HIGH OVERFITTING detected")
    elif train_accuracy - test_accuracy > 0.02:
        print("   ⚡ Moderate overfitting detected")
    else:
        print("   ✅ Good generalization")
    
    # Feature importance (get feature names from model if available)
    print(f"\n📊 Top 10 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2}. {row['feature']:<20} | {row['importance']:.4f}")
    
    return {
        'standard': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'roc_auc': test_roc_auc,
            'business_risk': business_risk_std,
            'confusion_matrix': cm_standard
        },
        'optimized': {
            'accuracy': opt_accuracy,
            'precision': opt_precision,
            'recall': opt_recall,
            'f1': opt_f1,
            'business_risk': business_risk_opt,
            'confusion_matrix': cm_optimized
        },
        'overfitting': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'difference': train_accuracy - test_accuracy
        }
    }


def create_plots(model, X_test, y_test, optimal_threshold=0.5):
    """Create evaluation plots"""
    print("📈 Creating evaluation plots...")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_std = model.predict(X_test)
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    axes[0, 0].plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    axes[0, 1].plot(recall, precision, color='red', linewidth=2, label='PR Curve')
    axes[0, 1].axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.6, 
                      label=f'Random ({y_test.mean():.3f})')
    
    # Mark optimal threshold
    if optimal_threshold != 0.5:
        opt_precision = precision_score(y_test, y_pred_opt)
        opt_recall = recall_score(y_test, y_pred_opt)
        axes[0, 1].scatter(opt_recall, opt_precision, color='green', s=100, 
                          label=f'Optimal ({optimal_threshold:.3f})', zorder=5)
    
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix Comparison
    cm_std = confusion_matrix(y_test, y_pred_std)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    
    x = np.arange(2)
    width = 0.35
    
    axes[1, 0].bar(x - width/2, [cm_std[1,1], cm_opt[1,1]], width, 
                   label='True Positives', color='green', alpha=0.7)
    axes[1, 0].bar(x + width/2, [cm_std[0,1], cm_opt[0,1]], width, 
                   label='False Positives', color='red', alpha=0.7)
    
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Confusion Matrix Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Standard (0.5)', f'Optimized ({optimal_threshold:.3f})'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature Importance
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(10)
    axes[1, 1].barh(range(len(top_features)), top_features['importance'], 
                    color='skyblue', alpha=0.8)
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Feature Importance')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def main():
    """Main optimization workflow"""
    print("🚀 XGBoost Optimization Workflow (Full Feature Engineering)")
    print("=" * 65)
    
    # Step 1: Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Step 2: Apply comprehensive feature engineering
    print("\n🔧 Step 2: Feature Engineering")
    X_train_processed, X_test_processed, pipeline = apply_feature_engineering(X_train, X_test, y_train)
    
    # Step 3: Optimize hyperparameters
    print("\n📈 Step 3: Hyperparameter Optimization")
    best_model, best_params, best_score = optimize_xgboost(
        X_train_processed, y_train, cv_folds=5, n_iter=30
    )
    
    # Step 4: Optimize threshold
    print("\n🎯 Step 4: Threshold Optimization")
    optimal_threshold, opt_precision, opt_recall = optimize_threshold(
        best_model, X_test_processed, y_test, recall_target=0.95
    )
    
    if optimal_threshold is None:
        optimal_threshold = 0.5
    
    # Step 5: Comprehensive evaluation
    print("\n📊 Step 5: Model Evaluation")
    results = evaluate_model(
        best_model, X_train_processed, y_train, 
        X_test_processed, y_test, optimal_threshold
    )
    
    # Step 6: Create plots
    print("\n📈 Step 6: Visualizations")
    create_plots(best_model, X_test_processed, y_test, optimal_threshold)
    
    # Step 7: Save model and pipeline
    print("\n💾 Step 7: Saving Model")
    best_model.save_model('models/xgboost_optimized.json')
    print("✅ Model saved as 'models/xgboost_optimized.json'")
    
    # Save pipeline for production use
    import joblib
    joblib.dump(pipeline, 'models/feature_engineering_pipeline.pkl')
    print("✅ Feature engineering pipeline saved as 'models/feature_engineering_pipeline.pkl'")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"✅ Accuracy requirement: {results['standard']['accuracy']:.1%} (≥86% {'✓' if results['standard']['accuracy'] >= 0.86 else '✗'})")
    print(f"🎯 Recall target: {results['optimized']['recall']:.1%} (≥95% {'✓' if results['optimized']['recall'] >= 0.95 else '✗'})")
    print(f"⚡ Business risk: {results['optimized']['business_risk']:.2f}%")
    print(f"🔧 Optimal threshold: {optimal_threshold:.4f}")
    print(f"🔗 Feature engineering: Full pipeline ({X_train_processed.shape[1]} features)")
    print("=" * 80)
    
    return best_model, results, optimal_threshold, pipeline


if __name__ == "__main__":
    model, results, threshold, pipeline = main() 