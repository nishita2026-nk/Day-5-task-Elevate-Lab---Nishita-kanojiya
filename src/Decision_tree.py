import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_PATH = "dataset/heart.csv"
OUTPUT_DIR = "outputs"
EXCEL_PATH = os.path.join(OUTPUT_DIR, "model_results.xlsx")
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. LOAD AND PREPARE DATA
# ==========================================================
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place heart.csv inside the dataset folder."
        )

    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError(
            "Expected target column named 'target'. "
            f"Available columns: {df.columns.tolist()}"
        )

    if df["target"].isna().any():
        raise ValueError("The target column contains missing values.")

    print("\nDATASET INFORMATION")
    print("=" * 55)
    print("Shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nTarget distribution:\n", df["target"].value_counts())

    return df


def preprocess_data(df):
    """Impute features and one-hot encode categorical columns."""
    X = df.drop(columns="target").copy()
    y = df["target"].copy()

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        mode = X[col].mode()
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        X[col] = X[col].fillna(fill_value)

    if categorical_cols:
        X = pd.get_dummies(
            X, columns=categorical_cols, drop_first=True, dtype=int
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    if y.nunique() != 2:
        raise ValueError(
            "This script's ROC/PR analysis expects a binary target. "
            f"Found {y.nunique()} classes: {sorted(y.unique().tolist())}"
        )

    return X, y


def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )


# ==========================================================
# 2. MODEL TRAINING AND EVALUATION
# ==========================================================
def evaluate_model(model, X_test, y_test, model_name, plot_filename):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, predictions)

    print(f"\n{model_name.upper()} RESULTS")
    print("=" * 55)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, plot_filename),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    return accuracy, report, cm


def visualize_decision_tree(model, feature_names):
    plt.figure(figsize=(24, 14))
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=[str(c) for c in model.classes_],
        filled=True, rounded=True, fontsize=8
    )
    plt.title("Decision Tree Classifier")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "decision_tree.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


# ==========================================================
# 3. DECISION TREE DEPTH / OVERFITTING ANALYSIS
# ==========================================================
def analyze_tree_depth(X_train, y_train):
    depths = list(range(1, 16))
    rows = []

    # Use CV on training data; do not select depth using the test set.
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    for depth in depths:
        model = DecisionTreeClassifier(
            max_depth=depth, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        cv_score = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        ).mean()

        rows.append({
            "max_depth": depth,
            "training_accuracy": train_score,
            "cv_accuracy": cv_score
        })

    results = pd.DataFrame(rows)
    best_depth = int(
        results.loc[results["cv_accuracy"].idxmax(), "max_depth"]
    )

    plt.figure(figsize=(9, 6))
    plt.plot(
        results["max_depth"], results["training_accuracy"],
        marker="o", label="Training Accuracy"
    )
    plt.plot(
        results["max_depth"], results["cv_accuracy"],
        marker="o", label="Cross-validation Accuracy"
    )
    plt.xlabel("Maximum Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Depth vs Accuracy")
    plt.xticks(depths)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "depth_vs_accuracy.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("\nBest Decision Tree depth (training CV):", best_depth)
    return best_depth, results


# ==========================================================
# 4. RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================================
def tune_random_forest(X_train, y_train):
    print("\nRANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 55)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )
    search.fit(X_train, y_train)

    tuning = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    selected_cols = [
        "param_n_estimators", "param_max_depth",
        "param_min_samples_split", "param_min_samples_leaf",
        "mean_train_score", "mean_test_score",
        "std_test_score", "rank_test_score"
    ]
    tuning = tuning[selected_cols].reset_index(drop=True)

    print("Best parameters:", search.best_params_)
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    top = tuning.head(10).copy()
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, len(top) + 1),
        top["mean_test_score"]
    )
    plt.xlabel("Top parameter combination (rank)")
    plt.ylabel("Mean CV Accuracy")
    plt.title("Random Forest Hyperparameter Tuning")
    plt.xticks(range(1, len(top) + 1))
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "hyperparameter_tuning.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    return search.best_estimator_, search.best_params_, tuning


# ==========================================================
# 5. FEATURE IMPORTANCE
# ==========================================================
def feature_importance(model, feature_names):
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    top = importance.head(15).sort_values("Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Importance")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    return importance


# ==========================================================
# 6. CROSS-VALIDATION COMPARISON
# ==========================================================
def run_cross_validation(dt_model, rf_model, X_train, y_train):
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    dt_scores = cross_val_score(
        dt_model, X_train, y_train, cv=cv,
        scoring="accuracy", n_jobs=-1
    )
    rf_scores = cross_val_score(
        rf_model, X_train, y_train, cv=cv,
        scoring="accuracy", n_jobs=-1
    )

    results = pd.DataFrame({
        "Fold": range(1, CV_FOLDS + 1),
        "Decision Tree Accuracy": dt_scores,
        "Tuned Random Forest Accuracy": rf_scores
    })

    plt.figure(figsize=(8, 5))
    plt.plot(results["Fold"], dt_scores, marker="o", label="Decision Tree")
    plt.plot(
        results["Fold"], rf_scores, marker="o",
        label="Tuned Random Forest"
    )
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Training-set 5-Fold Cross-Validation")
    plt.xticks(range(1, CV_FOLDS + 1))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "cross_validation.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    return results, dt_scores, rf_scores


# ==========================================================
# 7. ROC-AUC AND PRECISION-RECALL CURVES
# ==========================================================
def probability_metrics(models, X_test, y_test):
    classes = list(models[0][1].classes_)
    positive_class = classes[-1]

    roc_rows = []
    pr_rows = []

    plt.figure(figsize=(8, 6))
    for name, model in models:
        positive_index = list(model.classes_).index(positive_class)
        probabilities = model.predict_proba(X_test)[:, positive_index]

        # roc_auc_score() does not accept pos_label in current
        # scikit-learn versions. The model's probability column already
        # corresponds to positive_class, so pass y_test and probabilities.
        roc_score = roc_auc_score(y_test, probabilities)
        fpr, tpr, _ = roc_curve(
            y_test, probabilities, pos_label=positive_class
        )
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.3f})")
        roc_rows.append({"Model": name, "ROC-AUC": roc_score})

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (positive class: {positive_class})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "roc_curve.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, model in models:
        positive_index = list(model.classes_).index(positive_class)
        probabilities = model.predict_proba(X_test)[:, positive_index]
        precision, recall, _ = precision_recall_curve(
            y_test, probabilities, pos_label=positive_class
        )
        pr_score = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_score:.3f})")
        pr_rows.append({"Model": name, "PR-AUC": pr_score})

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (positive class: {positive_class})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "precision_recall_curve.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


# ==========================================================
# 8. MODEL COMPARISON AND RECOMMENDATION
# ==========================================================
def final_recommendation(comparison):
    # Accuracy is used as the primary selection metric; ROC-AUC is
    # included as a secondary metric for interpretation.
    best_idx = comparison["Accuracy"].idxmax()
    best = comparison.loc[best_idx]
    recommendation = str(best["Model"])

    reason = (
        "Highest test accuracy among the evaluated models. "
        "Review ROC-AUC, precision, recall and cross-validation "
        "before treating this as the final choice."
    )

    print("\nFINAL MODEL RECOMMENDATION")
    print("=" * 55)
    print("Recommended model:", recommendation)
    print(reason)

    return pd.DataFrame({
        "Metric": [
            "Recommended Model", "Selection Basis",
            "Test Accuracy", "ROC-AUC", "PR-AUC"
        ],
        "Result": [
            recommendation, reason, best["Accuracy"],
            best["ROC-AUC"], best["PR-AUC"]
        ]
    })


# ==========================================================
# 9. EXCEL REPORT
# ==========================================================
def save_excel_results(
    df, comparison, depth_results, importance, cv_results,
    reports, confusion_matrices, roc_results, pr_results,
    tuning_results, recommendation
):
    dataset_summary = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
        "Value": [
            df.shape[0], df.shape[1],
            int(df.isnull().sum().sum()),
            int(df.duplicated().sum())
        ]
    })

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Model Comparison", index=False)
        depth_results.to_excel(writer, sheet_name="Tree Depth Analysis", index=False)
        importance.to_excel(writer, sheet_name="Feature Importance", index=False)
        cv_results.to_excel(writer, sheet_name="Cross Validation", index=False)
        roc_results.to_excel(writer, sheet_name="ROC-AUC", index=False)
        pr_results.to_excel(writer, sheet_name="Precision Recall", index=False)
        tuning_results.to_excel(writer, sheet_name="Hyperparameter Tuning", index=False)
        recommendation.to_excel(writer, sheet_name="Final Recommendation", index=False)
        dataset_summary.to_excel(writer, sheet_name="Dataset Summary", index=False)
        df.head(20).to_excel(writer, sheet_name="Dataset Preview", index=False)

        for name, report in reports.items():
            pd.DataFrame(report).transpose().to_excel(
                writer, sheet_name=f"{name[:24]} Report"
            )

        for name, cm in confusion_matrices.items():
            labels = sorted(df["target"].unique())
            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual {v}" for v in labels],
                columns=[f"Predicted {v}" for v in labels]
            )
            cm_df.to_excel(writer, sheet_name=f"{name[:24]} Confusion")

        # Improve readability of every worksheet.
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max(max_length + 2, 12), 55)

    print(f"\nExcel report saved: {EXCEL_PATH}")


# ==========================================================
# 10. MAIN
# ==========================================================
def main():
    print("=" * 60)
    print("DECISION TREE AND RANDOM FOREST")
    print("=" * 60)

    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Baseline Decision Tree
    baseline_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline_dt.fit(X_train, y_train)
    dt_acc, dt_report, dt_cm = evaluate_model(
        baseline_dt, X_test, y_test,
        "Decision Tree", "decision_tree_confusion_matrix.png"
    )
    visualize_decision_tree(baseline_dt, X.columns)

    # Choose tree depth using training-only cross-validation.
    best_depth, depth_results = analyze_tree_depth(X_train, y_train)
    tuned_dt = DecisionTreeClassifier(
        max_depth=best_depth, random_state=RANDOM_STATE
    )
    tuned_dt.fit(X_train, y_train)
    tuned_dt_acc, tuned_dt_report, tuned_dt_cm = evaluate_model(
        tuned_dt, X_test, y_test,
        "Tuned Decision Tree", "tuned_decision_tree_confusion_matrix.png"
    )

    # Baseline Random Forest
    baseline_rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    )
    baseline_rf.fit(X_train, y_train)
    base_rf_acc, base_rf_report, base_rf_cm = evaluate_model(
        baseline_rf, X_test, y_test,
        "Random Forest", "random_forest_confusion_matrix.png"
    )

    # Tuned Random Forest
    tuned_rf, best_params, tuning_results = tune_random_forest(
        X_train, y_train
    )
    tuned_rf_acc, tuned_rf_report, tuned_rf_cm = evaluate_model(
        tuned_rf, X_test, y_test,
        "Tuned Random Forest", "tuned_random_forest_confusion_matrix.png"
    )

    importance = feature_importance(tuned_rf, X.columns)

    cv_results, dt_cv, rf_cv = run_cross_validation(
        tuned_dt, tuned_rf, X_train, y_train
    )

    evaluated_models = [
        ("Decision Tree", baseline_dt),
        ("Tuned Decision Tree", tuned_dt),
        ("Random Forest", baseline_rf),
        ("Tuned Random Forest", tuned_rf)
    ]
    roc_results, pr_results = probability_metrics(
        evaluated_models, X_test, y_test
    )

    accuracy_map = {
        "Decision Tree": dt_acc,
        "Tuned Decision Tree": tuned_dt_acc,
        "Random Forest": base_rf_acc,
        "Tuned Random Forest": tuned_rf_acc
    }

    comparison = pd.DataFrame({
        "Model": list(accuracy_map.keys()),
        "Accuracy": list(accuracy_map.values())
    }).merge(roc_results, on="Model").merge(pr_results, on="Model")

    recommendation = final_recommendation(comparison)

    reports = {
        "Decision Tree": dt_report,
        "Tuned Decision Tree": tuned_dt_report,
        "Random Forest": base_rf_report,
        "Tuned Random Forest": tuned_rf_report
    }
    confusion_matrices = {
        "DT": dt_cm,
        "Tuned DT": tuned_dt_cm,
        "RF": base_rf_cm,
        "Tuned RF": tuned_rf_cm
    }

    save_excel_results(
        df, comparison, depth_results, importance, cv_results,
        reports, confusion_matrices, roc_results, pr_results,
        tuning_results, recommendation
    )

    print("\nCROSS-VALIDATION SUMMARY")
    print(f"Decision Tree mean CV accuracy: {dt_cv.mean():.4f}")
    print(f"Tuned Random Forest mean CV accuracy: {rf_cv.mean():.4f}")
    print("\nBest Random Forest parameters:", best_params)

    print("\nGenerated files:")
    for filename in [
        "decision_tree.png",
        "decision_tree_confusion_matrix.png",
        "tuned_decision_tree_confusion_matrix.png",
        "depth_vs_accuracy.png",
        "random_forest_confusion_matrix.png",
        "tuned_random_forest_confusion_matrix.png",
        "feature_importance.png",
        "cross_validation.png",
        "hyperparameter_tuning.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "model_results.xlsx"
    ]:
        print(os.path.join(OUTPUT_DIR, filename))

    print("\nTASK 5 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
