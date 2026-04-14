import joblib
import os
import json
from datetime import datetime
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import your load function
from load_data import load

def compute_metrics(y_true, y_pred, y_proba):
    """Compute evaluation metrics"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

def save_metrics_to_json(metrics_dict, filepath):
    """Save metrics to JSON file for better readability"""
    # Convert numpy values to Python native types for JSON serialization
    serializable_metrics = {}
    for model_name, data in metrics_dict.items():
        serializable_metrics[model_name] = {
            "metrics": {k: float(v) for k, v in data["metrics"].items()},
            "timestamp_file": data["timestamp_file"],
            "latest_file": data["latest_file"],
            "timestamp": data.get("timestamp", "unknown")
        }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)

def create_model_card(model_name, metrics, timestamp):
    """Create a model card with metadata"""
    return {
        "model_name": model_name,
        "training_date": timestamp,
        "metrics": metrics,
        "version": "1.0",
        "framework": "scikit-learn",
        "created_by": "model_training_pipeline"
    }

def train():
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"TRAINING RUN: {timestamp}")
    print(f"{'='*60}\n")
    
    # Create artifacts directory with timestamp subdirectory
    base_artifacts_dir = Path("../artifacts")
    timestamp_dir = base_artifacts_dir / f"run_{timestamp}"
    latest_dir = base_artifacts_dir / "latest"
    
    # Create directories
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------
    # Load Data
    # ----------------------------
    
    print("📊 Loading data...")
    df, X_train, X_test, y_train, y_test, test_indices, RANDOM_STATE = load()
    del df  # Free up memory
    
    # ----------------------------
    # Load Preprocessor
    # ----------------------------
    
    print("🔧 Loading preprocessor...")
    preprocessor = joblib.load("../artifacts/preprocessor.pkl")
    
    # ----------------------------
    # Define Models
    # ----------------------------
    
    print("🤖 Initializing models...")
    models = {
        "logistic_regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1])),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False
        )
    }
    
    # ----------------------------
    # Training + Evaluation
    # ----------------------------
    
    results = {}
    model_cards = {}
    
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"{'='*60}")
    
    for name, model in models.items():
        print(f"\n🎯 Training {name.replace('_', ' ').title()}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)
        
        # Print metrics
        print("📈 Metrics:")
        for k, v in metrics.items():
            print(f"  • {k}: {v:.4f}")
        
        # Save model with timestamp
        timestamp_filename = f"model_{name}_{timestamp}.pkl"
        timestamp_filepath = timestamp_dir / timestamp_filename
        
        # Save model in timestamp directory
        joblib.dump(pipeline, timestamp_filepath)
        
        # Also save in latest directory
        latest_filename = f"model_{name}_latest.pkl"
        latest_filepath = latest_dir / latest_filename
        joblib.dump(pipeline, latest_filepath)
        
        print(f"💾 Model saved:")
        print(f"  • Versioned: {timestamp_filepath}")
        print(f"  • Latest: {latest_filepath}")
        
        # Store results
        results[name] = {
            "model": pipeline,
            "metrics": metrics,
            "timestamp_file": str(timestamp_filepath),
            "latest_file": str(latest_filepath),
            "timestamp": timestamp
        }
        
        # Create model card
        model_cards[name] = create_model_card(name, metrics, timestamp)
    
    # ----------------------------
    # Save Results and Comparison
    # ----------------------------
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Print comparison table
    print("\n📊 Model Performance Comparison:")
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}")
    print("-" * 75)
    
    for name, result in results.items():
        metrics = result["metrics"]
        print(f"{name.replace('_', ' ').title():<25} "
              f"{metrics['accuracy']:.4f}     "
              f"{metrics['precision']:.4f}     "
              f"{metrics['recall']:.4f}     "
              f"{metrics['f1']:.4f}     "
              f"{metrics['roc_auc']:.4f}")
    
    # ----------------------------
    # Identify and Save Best Model
    # ----------------------------
    
    # Best by recall (business priority)
    best_recall_name = max(results, key=lambda x: results[x]["metrics"]["recall"])
    best_recall_pipeline = results[best_recall_name]["model"]
    
    # Save best recall model
    best_recall_timestamp = f"model_best_recall_{timestamp}.pkl"
    best_recall_latest = "model_best_recall_latest.pkl"
    
    joblib.dump(best_recall_pipeline, timestamp_dir / best_recall_timestamp)
    joblib.dump(best_recall_pipeline, latest_dir / best_recall_latest)
    
    print(f"\n🏆 Best Model by Recall:")
    print(f"  • Model: {best_recall_name.replace('_', ' ').title()}")
    print(f"  • Recall: {results[best_recall_name]['metrics']['recall']:.4f}")
    print(f"  • Saved as: {best_recall_timestamp}")
    
    # Best by F1 score (balanced metric)
    best_f1_name = max(results, key=lambda x: results[x]["metrics"]["f1"])
    if best_f1_name != best_recall_name:
        best_f1_pipeline = results[best_f1_name]["model"]
        best_f1_timestamp = f"model_best_f1_{timestamp}.pkl"
        best_f1_latest = "model_best_f1_latest.pkl"
        
        joblib.dump(best_f1_pipeline, timestamp_dir / best_f1_timestamp)
        joblib.dump(best_f1_pipeline, latest_dir / best_f1_latest)
        
        print(f"\n🥈 Best Model by F1 Score:")
        print(f"  • Model: {best_f1_name.replace('_', ' ').title()}")
        print(f"  • F1 Score: {results[best_f1_name]['metrics']['f1']:.4f}")
        print(f"  • Saved as: {best_f1_timestamp}")
    
    # Best by ROC AUC
    best_auc_name = max(results, key=lambda x: results[x]["metrics"]["roc_auc"])
    if best_auc_name not in [best_recall_name, best_f1_name]:
        best_auc_pipeline = results[best_auc_name]["model"]
        best_auc_timestamp = f"model_best_auc_{timestamp}.pkl"
        best_auc_latest = "model_best_auc_latest.pkl"
        
        joblib.dump(best_auc_pipeline, timestamp_dir / best_auc_timestamp)
        joblib.dump(best_auc_pipeline, latest_dir / best_auc_latest)
        
        print(f"\n📈 Best Model by ROC AUC:")
        print(f"  • Model: {best_auc_name.replace('_', ' ').title()}")
        print(f"  • ROC AUC: {results[best_auc_name]['metrics']['roc_auc']:.4f}")
        print(f"  • Saved as: {best_auc_timestamp}")
    
    # ----------------------------
    # Prepare Metrics Summary for JSON
    # ----------------------------
    
    # Create metrics summary dictionary
    metrics_summary = {}
    for name, data in results.items():
        metrics_summary[name] = {
            "metrics": data["metrics"],
            "timestamp_file": data["timestamp_file"],
            "latest_file": data["latest_file"],
            "timestamp": data["timestamp"]
        }
    
    # ----------------------------
    # Save Metadata and Summary
    # ----------------------------
    
    # Save results summary
    summary = {
        "training_timestamp": timestamp,
        "best_recall_model": {
            "name": best_recall_name,
            "metrics": {k: float(v) for k, v in results[best_recall_name]["metrics"].items()},
            "timestamp_file": best_recall_timestamp,
            "latest_file": best_recall_latest
        },
        "best_f1_model": {
            "name": best_f1_name,
            "metrics": {k: float(v) for k, v in results[best_f1_name]["metrics"].items()},
            "timestamp_file": f"model_best_f1_{timestamp}.pkl" if best_f1_name != best_recall_name else "same_as_best_recall",
            "latest_file": "model_best_f1_latest.pkl" if best_f1_name != best_recall_name else "same_as_best_recall"
        },
        "best_auc_model": {
            "name": best_auc_name,
            "metrics": {k: float(v) for k, v in results[best_auc_name]["metrics"].items()},
            "timestamp_file": f"model_best_auc_{timestamp}.pkl" if best_auc_name not in [best_recall_name, best_f1_name] else "same_as_other_best",
            "latest_file": "model_best_auc_latest.pkl" if best_auc_name not in [best_recall_name, best_f1_name] else "same_as_other_best"
        },
        "all_models": {
            name: {
                "metrics": {k: float(v) for k, v in data["metrics"].items()},
                "timestamp_file": data["timestamp_file"],
                "latest_file": data["latest_file"]
            }
            for name, data in results.items()
        },
        "data_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "random_state": RANDOM_STATE
        }
    }
    
    # Save summary to JSON
    summary_file = timestamp_dir / f"training_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Also save to latest directory
    latest_summary = latest_dir / "training_summary_latest.json"
    with open(latest_summary, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save model cards
    model_cards_file = timestamp_dir / f"model_cards_{timestamp}.json"
    with open(model_cards_file, 'w') as f:
        json.dump(model_cards, f, indent=4)
    
    # Save metrics separately
    save_metrics_to_json(metrics_summary, timestamp_dir / f"metrics_{timestamp}.json")
    save_metrics_to_json(metrics_summary, latest_dir / "metrics_latest.json")
    
    # ----------------------------
    # Final Output
    # ----------------------------
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE! ✅")
    print(f"{'='*60}")
    print(f"\n📁 Artifacts saved in:")
    print(f"  • Timestamped directory: {timestamp_dir}")
    print(f"  • Latest directory: {latest_dir}")
    
    print(f"\n📄 Summary files created:")
    print(f"  • training_summary_{timestamp}.json")
    print(f"  • model_cards_{timestamp}.json")
    print(f"  • metrics_{timestamp}.json")
    
    print(f"\n🤖 Models saved:")
    for name in results.keys():
        print(f"  • {name.replace('_', ' ').title()}:")
        print(f"      - model_{name}_{timestamp}.pkl")
        print(f"      - model_{name}_latest.pkl")
    
    print(f"\n🏆 Best models:")
    print(f"  • Best Recall: model_best_recall_{timestamp}.pkl")
    if best_f1_name != best_recall_name:
        print(f"  • Best F1: model_best_f1_{timestamp}.pkl")
    if best_auc_name not in [best_recall_name, best_f1_name]:
        print(f"  • Best AUC: model_best_auc_{timestamp}.pkl")
    
    return results

if __name__ == "__main__":
    train()