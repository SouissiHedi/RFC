import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, fbeta_score
from load_data import load

def tune():
    # ─── Chargement ───────────────────────────────────────────────────────────────
    df, X_train, X_test, y_train, y_test, RANDOM_STATE = load()
    model = joblib.load("../artifacts/best_model.pkl")

    y_proba = model.predict_proba(X_test)[:, 1]

    # ─── Precision-Recall Curve + AP ──────────────────────────────────────────────
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='royalblue', lw=2,
            label=f'AP = {ap:.4f}')
    plt.xlabel("Recall (Failures Caught)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve – Test set", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.show()

    print(f"Average Precision (AP) = {ap:.4f}\n")

    # ─── Évaluation multi-seuils ──────────────────────────────────────────────────
    results = []

    for t in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        recall_val    = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1            = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        f2            = fbeta_score(y_test, y_pred, beta=2)

        results.append({
            'threshold': round(t, 2),
            'recall':    round(recall_val, 3),
            'precision': round(precision_val, 3),
            'f1':        round(f1, 3),
            'f2':        round(f2, 3),
            'fp':        fp,
            'fn':        fn,
            'alerts':    tp + fp   # nombre total d'alertes déclenchées
        })

    df_results = pd.DataFrame(results)

    # Affichage trié + mise en forme sympa
    print("Seuil   Recall   Precision   F1    F2     FP    FN    Alerts")
    print("-" * 65)
    for _, row in df_results.sort_values('threshold').iterrows():
        print(f"{row['threshold']:5.2f}   {row['recall']:6.3f}   {row['precision']:8.3f}   "
        f"{row['f1']:5.3f}  {row['f2']:5.3f}   {int(row['fp']):4d}  {int(row['fn']):4d}   {int(row['alerts']):6d}")

    # ─── Meilleurs seuils selon différents critères ───────────────────────────────
    print("\nMeilleurs seuils selon différents objectifs :")
    best_f1  = df_results.loc[df_results['f1'].idxmax()]
    best_f2  = df_results.loc[df_results['f2'].idxmax()]
    best_bal = df_results.loc[(df_results['precision'] >= 0.80) & (df_results['recall'] >= 0.70)].sort_values('f1', ascending=False).head(1)

    print(f"→ Meilleur F1           : seuil = {best_f1['threshold']:.2f} → Recall {best_f1['recall']:.3f} | Prec {best_f1['precision']:.3f} | F1 {best_f1['f1']:.3f}")
    print(f"→ Meilleur F2 (poids FN): seuil = {best_f2['threshold']:.2f} → Recall {best_f2['recall']:.3f} | Prec {best_f2['precision']:.3f} | F2 {best_f2['f2']:.3f}")

    if not best_bal.empty:
        b = best_bal.iloc[0]
        print(f"→ Bon compromis (prec≥80% & recall≥70%) : seuil = {b['threshold']:.2f} → "
            f"Recall {b['recall']:.3f} | Prec {b['precision']:.3f}")
    else:
        print("→ Aucun seuil ne satisfait prec ≥ 80% ET recall ≥ 70%")

if __name__ == "__main__":
    tune()