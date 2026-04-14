import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score, 
    confusion_matrix, 
    fbeta_score,
    roc_curve,
    auc,
    classification_report
)
from load_data import load
import warnings
warnings.filterwarnings('ignore')

# Configuration
OPTIMIZATION_METRIC = "recall"
MODELS_DIR = Path("../artifacts/latest")
ARTIFACTS_DIR = Path("../artifacts")

def load_all_models():
    """Load all trained models from the artifacts directory"""
    models = {}
    
    # Look for model files in latest directory
    if MODELS_DIR.exists():
        model_files = list(MODELS_DIR.glob("model_*.pkl"))
        
        for model_file in model_files:
            model_name = model_file.stem.replace("model_", "").replace("_latest", "")
            if "best" not in model_name:  # Skip best model files, we'll get individual models
                try:
                    models[model_name] = joblib.load(model_file)
                    print(f"✓ Loaded model: {model_name}")
                except Exception as e:
                    print(f"✗ Failed to load {model_file.name}: {e}")
    
    # If no models in latest, check timestamped directories
    if not models:
        timestamp_dirs = list(ARTIFACTS_DIR.glob("run_*"))
        if timestamp_dirs:
            latest_run = max(timestamp_dirs)
            model_files = list(latest_run.glob("model_*.pkl"))
            
            for model_file in model_files:
                model_name = model_file.stem.split("_")[-2]  # Get model name from timestamped filename
                if "best" not in model_name:
                    try:
                        models[model_name] = joblib.load(model_file)
                        print(f"✓ Loaded model from {latest_run.name}: {model_name}")
                    except Exception as e:
                        print(f"✗ Failed to load {model_file.name}: {e}")
    
    return models

def compute_all_metrics(y_true, y_pred, y_proba):
    """Compute comprehensive metrics for model evaluation"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1': 2 * (tp / (tp + fn)) * (tp / (tp + fp)) / ((tp / (tp + fn)) + (tp / (tp + fp))) if (tp + fn) > 0 and (tp + fp) > 0 else 0,
        'f2': fbeta_score(y_true, y_pred, beta=2),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'tn': tn,
        'total_alerts': tp + fp,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    
    return metrics

def evaluate_thresholds(y_true, y_proba, model_name):
    """Evaluate model performance at different thresholds"""
    results = []
    
    # Use more granular thresholds
    thresholds = np.arange(0.01, 0.99, 0.02)
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        
        results.append({
            'model': model_name,
            'threshold': round(t, 3),
            **metrics
        })
    
    return pd.DataFrame(results)

def find_optimal_thresholds(df_results, model_name):
    """Find optimal thresholds for different business objectives"""
    if df_results.empty:
        return {}
    
    # Filter out extreme thresholds
    df_filtered = df_results[df_results['threshold'].between(0.05, 0.95)]
    
    if df_filtered.empty:
        df_filtered = df_results
    
    optimals = {
        'model': model_name,
        'best_f1': df_filtered.loc[df_filtered['f1'].idxmax()].to_dict(),
        'best_f2': df_filtered.loc[df_filtered['f2'].idxmax()].to_dict(),
        'best_recall': df_filtered.loc[df_filtered['recall'].idxmax()].to_dict(),
        'best_precision': df_filtered.loc[df_filtered['precision'].idxmax()].to_dict(),
        'balanced_080_070': None,  # precision ≥ 0.80, recall ≥ 0.70
        'balanced_075_075': None,  # precision ≥ 0.75, recall ≥ 0.75
        'balanced_085_065': None   # precision ≥ 0.85, recall ≥ 0.65
    }
    
    # Find balanced thresholds
    balanced_criteria = [
        ('balanced_080_070', 0.80, 0.70),
        ('balanced_075_075', 0.75, 0.75),
        ('balanced_085_065', 0.85, 0.65)
    ]
    
    for key, min_precision, min_recall in balanced_criteria:
        filtered = df_filtered[
            (df_filtered['precision'] >= min_precision) &
            (df_filtered['recall'] >= min_recall)
        ]
        
        if not filtered.empty:
            optimals[key] = filtered.sort_values('f1', ascending=False).iloc[0].to_dict()
    
    return optimals

def plot_comparison_figure(all_results, model_performance):
    """Create comparison plots for all models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Precision-Recall Curves
    ax = axes[0]
    for model_name, df in all_results.items():
        # Get unique precision-recall pairs
        df_sorted = df.sort_values('threshold')
        ax.plot(df_sorted['recall'], df_sorted['precision'], 
                label=f"{model_name}", linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (All Models)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 2. F1-Score vs Threshold
    ax = axes[1]
    for model_name, df in all_results.items():
        df_sorted = df.sort_values('threshold')
        ax.plot(df_sorted['threshold'], df_sorted['f1'], 
                label=f"{model_name}", linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Recall vs Threshold
    ax = axes[2]
    for model_name, df in all_results.items():
        df_sorted = df.sort_values('threshold')
        ax.plot(df_sorted['threshold'], df_sorted['recall'], 
                label=f"{model_name}", linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Precision vs Threshold
    ax = axes[3]
    for model_name, df in all_results.items():
        df_sorted = df.sort_values('threshold')
        ax.plot(df_sorted['threshold'], df_sorted['precision'], 
                label=f"{model_name}", linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Model Performance Comparison (Bar chart)
    ax = axes[4]
    performance_df = pd.DataFrame(model_performance).T
    metrics_to_plot = ['f1', 'recall', 'precision', 'accuracy']
    
    x = np.arange(len(performance_df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i*width, performance_df[metric], width, label=metric)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(performance_df.index, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Trade-off: Recall vs Precision at optimal F1
    ax = axes[5]
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    
    for idx, (model_name, df) in enumerate(all_results.items()):
        best_f1_idx = df['f1'].idxmax()
        best_row = df.loc[best_f1_idx]
        
        ax.scatter(best_row['recall'], best_row['precision'], 
                  s=150, marker=markers[idx % len(markers)], 
                  label=f"{model_name}", edgecolors='black', linewidth=1)
        
        # Add threshold value as text
        ax.annotate(f"th={best_row['threshold']:.2f}", 
                   (best_row['recall'], best_row['precision']),
                   textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=9)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Optimal F1 Points (Recall vs Precision)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def print_model_summary_table(model_performance):
    """Print a comprehensive summary table of all models"""
    print("\n" + "="*120)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*120)
    
    # Create DataFrame for display
    summary_df = pd.DataFrame(model_performance).T
    
    # Format for display
    display_df = summary_df.copy()
    for col in ['f1', 'recall', 'precision', 'accuracy', 'f2', 'specificity']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string())
    print("="*120)

def print_detailed_threshold_analysis(optimal_thresholds):
    """Print detailed threshold analysis for each model"""
    print("\n" + "="*120)
    print("OPTIMAL THRESHOLDS ANALYSIS")
    print("="*120)
    
    for model_name, thresholds in optimal_thresholds.items():
        print(f"\n📊 Model: {model_name.upper()}")
        print("-"*80)
        
        if thresholds.get('best_f1'):
            best = thresholds['best_f1']
            print(f"🎯 Best F1-score (threshold={best['threshold']:.3f}):")
            print(f"   • Recall:    {best['recall']:.3f}")
            print(f"   • Precision: {best['precision']:.3f}")
            print(f"   • F1-score:  {best['f1']:.3f}")
            print(f"   • Alerts:    {best['total_alerts']} (FP: {best['fp']}, TP: {best['tp']})")
        
        if thresholds.get('best_f2'):
            best = thresholds['best_f2']
            print(f"\n⚖️  Best F2-score (threshold={best['threshold']:.3f}):")
            print(f"   • Recall:    {best['recall']:.3f}")
            print(f"   • Precision: {best['precision']:.3f}")
            print(f"   • F2-score:  {best['f2']:.3f}")
        
        # Show balanced thresholds
        balanced_keys = [k for k in thresholds.keys() if 'balanced' in k]
        for key in balanced_keys:
            if thresholds[key]:
                best = thresholds[key]
                _, min_prec, min_rec = key.split('_')
                min_prec = float(f"0.{min_prec}")
                min_rec = float(f"0.{min_rec}")
                
                print(f"\n✅ Balanced (Prec≥{min_prec:.0%}, Rec≥{min_rec:.0%}) (threshold={best['threshold']:.3f}):")
                print(f"   • Recall:    {best['recall']:.3f}")
                print(f"   • Precision: {best['precision']:.3f}")
                print(f"   • F1-score:  {best['f1']:.3f}")
        
        print("-"*80)

def print_business_recommendations(optimal_thresholds, test_size):
    """Print business-oriented recommendations"""
    print("\n" + "="*120)
    print("BUSINESS RECOMMENDATIONS")
    print("="*120)
    
    recommendations = []
    
    for model_name, thresholds in optimal_thresholds.items():
        if thresholds.get('balanced_080_070'):
            best = thresholds['balanced_080_070']
            rec = {
                'model': model_name,
                'threshold': best['threshold'],
                'recall': best['recall'],
                'precision': best['precision'],
                'f1': best['f1'],
                'alerts_per_1000': (best['total_alerts'] / test_size) * 1000,
                'missed_failures': best['fn']
            }
            recommendations.append(rec)
    
    if recommendations:
        # Sort by F1 score
        recommendations.sort(key=lambda x: x['f1'], reverse=True)
        
        print("\n🎯 Recommended Operating Points (Precision ≥ 80%, Recall ≥ 70%):")
        print("-"*100)
        print(f"{'Model':<20} {'Threshold':<10} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Alerts/1000':<12} {'Missed':<10}")
        print("-"*100)
        
        for rec in recommendations:
            print(f"{rec['model']:<20} {rec['threshold']:<10.3f} {rec['recall']:<10.3f} "
                  f"{rec['precision']:<12.3f} {rec['f1']:<10.3f} "
                  f"{rec['alerts_per_1000']:<12.1f} {rec['missed_failures']:<10}")
        
        print("-"*100)
        
        # Best overall recommendation
        best_rec = recommendations[0]
        print(f"\n🏆 TOP RECOMMENDATION: {best_rec['model'].upper()}")
        print(f"   • Operating threshold: {best_rec['threshold']:.3f}")
        print(f"   • Expected performance:")
        print(f"     - Detects {best_rec['recall']:.1%} of actual failures")
        print(f"     - {best_rec['precision']:.1%} of alerts are true failures")
        print(f"     - Balanced F1-score: {best_rec['f1']:.3f}")
        print(f"     - Estimated alerts: {best_rec['alerts_per_1000']:.1f} per 1000 units")
        print(f"     - Missed failures: {best_rec['missed_failures']} (needs manual review)")
    else:
        print("\n⚠️  No models meet the minimum requirements (Precision ≥ 80%, Recall ≥ 70%)")
        print("   Consider relaxing requirements or improving model performance.")

def tune_all_models():
    """Main function to tune and evaluate all models"""
    
    # Load data
    print("📊 Loading data...")
    df, X_train, X_test, y_train, y_test, test_indices, RANDOM_STATE = load()
    test_size = len(X_test)
    
    # Load all models
    print("\n🤖 Loading trained models...")
    models = load_all_models()
    
    if not models:
        print("❌ No models found in artifacts directory!")
        return
    
    print(f"\n✅ Loaded {len(models)} models for evaluation")
    
    # Evaluate each model
    all_results = {}
    model_performance = {}
    optimal_thresholds = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate at different thresholds
        df_results = evaluate_thresholds(y_test, y_proba, model_name)
        all_results[model_name] = df_results
        
        # Find optimal thresholds
        optimals = find_optimal_thresholds(df_results, model_name)
        optimal_thresholds[model_name] = optimals
        
        # Default threshold (0.5) performance
        y_pred_default = model.predict(X_test)
        default_metrics = compute_all_metrics(y_test, y_pred_default, y_proba)
        model_performance[model_name] = default_metrics
        
        # Print model-specific summary
        print(f"\nDefault threshold (0.5) performance:")
        print(f"  • Accuracy:  {default_metrics['accuracy']:.4f}")
        print(f"  • Recall:    {default_metrics['recall']:.4f}")
        print(f"  • Precision: {default_metrics['precision']:.4f}")
        print(f"  • F1-score:  {default_metrics['f1']:.4f}")
        
        if optimals.get('best_f1'):
            best = optimals['best_f1']
            print(f"\nOptimal F1 threshold ({best['threshold']:.3f}):")
            print(f"  • F1-score:  {best['f1']:.4f} (improvement: {best['f1'] - default_metrics['f1']:+.4f})")
            print(f"  • Recall:    {best['recall']:.4f}")
            print(f"  • Precision: {best['precision']:.4f}")
    
    # Create visual comparisons
    print(f"\n📈 Generating comparison plots...")
    plot_comparison_figure(all_results, model_performance)
    
    # Print comprehensive summaries
    print_model_summary_table(model_performance)
    print_detailed_threshold_analysis(optimal_thresholds)
    print_business_recommendations(optimal_thresholds, test_size)
    
    # Save results
    save_results(all_results, model_performance, optimal_thresholds)
    
    return all_results, model_performance, optimal_thresholds

def save_results(all_results, model_performance, optimal_thresholds):
    """Save analysis results to files"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"../artifacts/threshold_analysis_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all results DataFrames
    for model_name, df in all_results.items():
        df.to_csv(output_dir / f"{model_name}_threshold_analysis.csv", index=False)
    
    # Save model performance
    perf_df = pd.DataFrame(model_performance).T
    perf_df.to_csv(output_dir / "model_performance_summary.csv")
    
    # Save optimal thresholds
    optimal_df = pd.DataFrame({
        model: {k: v.get('threshold', np.nan) if isinstance(v, dict) else v 
                for k, v in thresholds.items() if 'threshold' in str(v)}
        for model, thresholds in optimal_thresholds.items()
    }).T
    optimal_df.to_csv(output_dir / "optimal_thresholds.csv")
    
    # Save detailed analysis report
    with open(output_dir / "analysis_report.txt", 'w') as f:
        f.write("MODEL THRESHOLD ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        for model_name in all_results.keys():
            f.write(f"\n{model_name.upper()}\n")
            f.write("-"*30 + "\n")
            
            if model_name in optimal_thresholds:
                thresholds = optimal_thresholds[model_name]
                if thresholds.get('best_f1'):
                    best = thresholds['best_f1']
                    f.write(f"Best F1: threshold={best['threshold']:.3f}, "
                           f"recall={best['recall']:.3f}, "
                           f"precision={best['precision']:.3f}, "
                           f"f1={best['f1']:.3f}\n")
    
    print(f"\n💾 Results saved to: {output_dir}")

if __name__ == "__main__":
    tune_all_models()