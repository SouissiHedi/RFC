import joblib
from load_data import load

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train():
    # ----------------------------
    # Load Data
    # ----------------------------
    df, X_train, X_test, y_train, y_test ,RANDOM_STATE= load()
    del(df)
    # ----------------------------
    # Load Preprocessor
    # ----------------------------
    preprocessor = joblib.load("../artifacts/preprocessor.pkl")

    # ----------------------------
    # Define Models
    # ----------------------------

    models = {
        "Logistic_Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1])),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
    }

    # ----------------------------
    # Training + Evaluation
    # ----------------------------

    results = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name}")

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        roc = roc_auc_score(y_test, y_proba)
        print("ROC-AUC:", roc)

        # Store for comparison
        results[name] = {
            "model": pipeline,
            "roc_auc": roc
        }

    # ----------------------------
    # Save Best Model
    # ----------------------------
    best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
    best_pipeline = results[best_model_name]["model"]

    joblib.dump(best_pipeline, "../artifacts/best_model.pkl")

    print(f"\n🏆 Best model saved: {best_model_name}")

if __name__ == "__main__":
    train()