import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "dataset/heart.csv"
OUTPUT_DIR = "outputs"
EXCEL_PATH = os.path.join(OUTPUT_DIR, "model_results.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def preprocess_data(df):
    if "target" not in df.columns:
        raise ValueError("Target column 'target' was not found in the dataset.")

    X = df.drop(columns=["target"]).copy()
    y = df["target"].copy()

    numerical_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    print("\nNumerical Columns:", numerical_columns)
    print("\nCategorical Columns:", categorical_columns)

    if numerical_columns:
        X[numerical_columns] = X[numerical_columns].fillna(
            X[numerical_columns].median()
        )

    for column in categorical_columns:
        X[column] = X[column].fillna(X[column].mode()[0])

    if categorical_columns:
        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=True,
            dtype=int
        )

    print("\nProcessed Feature Shape:", X.shape)
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_decision_tree(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 50)
    print("DECISION TREE RESULTS")
    print("=" * 50)
    print("Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/decision_tree_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return accuracy, report, cm


def visualize_decision_tree(model, X):
    plt.figure(figsize=(24, 14))

    plot_tree(
        model,
        feature_names=X.columns,
        class_names=[str(c) for c in sorted(model.classes_)],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("Decision Tree Classifier")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/decision_tree.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nDecision Tree visualization saved.")


def analyze_tree_depth(X_train, X_test, y_train, y_test):
    depths = range(1, 16)
    training_accuracy = []
    testing_accuracy = []

    for depth in depths:
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        training_accuracy.append(model.score(X_train, y_train))
        testing_accuracy.append(model.score(X_test, y_test))

    results = pd.DataFrame({
        "max_depth": list(depths),
        "training_accuracy": training_accuracy,
        "testing_accuracy": testing_accuracy
    })

    print("\n" + "=" * 50)
    print("TREE DEPTH / OVERFITTING ANALYSIS")
    print("=" * 50)
    print(results)

    plt.figure(figsize=(9, 6))
    plt.plot(
        depths,
        training_accuracy,
        marker="o",
        label="Training Accuracy"
    )
    plt.plot(
        depths,
        testing_accuracy,
        marker="o",
        label="Testing Accuracy"
    )
    plt.xlabel("Maximum Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Depth vs Accuracy")
    plt.xticks(list(depths))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/depth_vs_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    best_depth = int(
        results.loc[
            results["testing_accuracy"].idxmax(),
            "max_depth"
        ]
    )

    print("\nBest Tree Depth:", best_depth)
    return best_depth, results


def train_tuned_tree(X_train, y_train, max_depth):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_random_forest(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)
    print("Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/random_forest_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return accuracy, report, cm


def feature_importance(model, X):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    print(importance)

    top_features = importance.head(15).sort_values(by="Importance")

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return importance


def cross_validation(decision_tree, random_forest, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    dt_scores = cross_val_score(
        decision_tree,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    rf_scores = cross_val_score(
        random_forest,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n" + "=" * 50)
    print("5-FOLD CROSS VALIDATION")
    print("=" * 50)
    print("\nDecision Tree CV Scores:", np.round(dt_scores, 4))
    print("Decision Tree Mean:", round(dt_scores.mean(), 4))
    print("\nRandom Forest CV Scores:", np.round(rf_scores, 4))
    print("Random Forest Mean:", round(rf_scores.mean(), 4))

    cv_results = pd.DataFrame({
        "Fold": range(1, 6),
        "Decision Tree Accuracy": dt_scores,
        "Random Forest Accuracy": rf_scores
    })

    folds = range(1, 6)

    plt.figure(figsize=(8, 5))
    plt.plot(
        folds,
        dt_scores,
        marker="o",
        label="Decision Tree"
    )
    plt.plot(
        folds,
        rf_scores,
        marker="o",
        label="Random Forest"
    )
    plt.xlabel("Cross-Validation Fold")
    plt.ylabel("Accuracy")
    plt.title("5-Fold Cross-Validation Accuracy")
    plt.xticks(list(folds))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/cross_validation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return dt_scores, rf_scores, cv_results


def compare_models(dt_accuracy, tuned_dt_accuracy, rf_accuracy):
    results = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Tuned Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            dt_accuracy,
            tuned_dt_accuracy,
            rf_accuracy
        ]
    })

    print("\n" + "=" * 50)
    print("MODEL ACCURACY COMPARISON")
    print("=" * 50)
    print(results)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=results,
        x="Model",
        y="Accuracy"
    )
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return results


def save_excel_results(
    df,
    results,
    depth_results,
    feature_importance_results,
    cv_results,
    dt_report,
    rf_report,
    dt_cm,
    rf_cm
):
    dt_report_df = pd.DataFrame(dt_report).transpose()
    rf_report_df = pd.DataFrame(rf_report).transpose()

    classes = sorted(df["target"].unique())

    dt_cm_df = pd.DataFrame(
        dt_cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes]
    )

    rf_cm_df = pd.DataFrame(
        rf_cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes]
    )

    dataset_summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Values",
            "Duplicate Rows"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            int(df.isnull().sum().sum()),
            int(df.duplicated().sum())
        ]
    })

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl"
    ) as writer:

        results.to_excel(
            writer,
            sheet_name="Model Comparison",
            index=False
        )

        depth_results.to_excel(
            writer,
            sheet_name="Tree Depth Analysis",
            index=False
        )

        feature_importance_results.to_excel(
            writer,
            sheet_name="Feature Importance",
            index=False
        )

        cv_results.to_excel(
            writer,
            sheet_name="Cross Validation",
            index=False
        )

        dt_report_df.to_excel(
            writer,
            sheet_name="DT Classification Report"
        )

        rf_report_df.to_excel(
            writer,
            sheet_name="RF Classification Report"
        )

        dt_cm_df.to_excel(
            writer,
            sheet_name="DT Confusion Matrix"
        )

        rf_cm_df.to_excel(
            writer,
            sheet_name="RF Confusion Matrix"
        )

        dataset_summary.to_excel(
            writer,
            sheet_name="Dataset Summary",
            index=False
        )

        df.head(20).to_excel(
            writer,
            sheet_name="Dataset Preview",
            index=False
        )

    print("\n" + "=" * 50)
    print("EXCEL RESULTS SAVED")
    print("=" * 50)
    print(f"Excel file: {EXCEL_PATH}")


def main():

    print("\n" + "=" * 60)
    print("DECISION TREE AND RANDOM FOREST")
    print("ELEVATE LABS - TASK 5")
    print("=" * 60)

    df = load_data()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Decision Tree
    decision_tree = train_decision_tree(
        X_train,
        y_train
    )

    dt_accuracy, dt_report, dt_cm = evaluate_decision_tree(
        decision_tree,
        X_test,
        y_test
    )

    visualize_decision_tree(
        decision_tree,
        X
    )

    # Overfitting analysis
    best_depth, depth_results = analyze_tree_depth(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Tuned Decision Tree
    tuned_tree = train_tuned_tree(
        X_train,
        y_train,
        best_depth
    )

    tuned_predictions = tuned_tree.predict(
        X_test
    )

    tuned_dt_accuracy = accuracy_score(
        y_test,
        tuned_predictions
    )

    print(
        "\nTuned Decision Tree Accuracy:",
        round(tuned_dt_accuracy, 4)
    )

    # Random Forest
    random_forest = train_random_forest(
        X_train,
        y_train
    )

    rf_accuracy, rf_report, rf_cm = evaluate_random_forest(
        random_forest,
        X_test,
        y_test
    )

    # Feature importance
    importance_results = feature_importance(
        random_forest,
        X
    )

    # Cross validation
    dt_cv_scores, rf_cv_scores, cv_results = cross_validation(
        tuned_tree,
        random_forest,
        X,
        y
    )

    # Model comparison
    results = compare_models(
        dt_accuracy,
        tuned_dt_accuracy,
        rf_accuracy
    )

    # Excel results
    save_excel_results(
        df=df,
        results=results,
        depth_results=depth_results,
        feature_importance_results=importance_results,
        cv_results=cv_results,
        dt_report=dt_report,
        rf_report=rf_report,
        dt_cm=dt_cm,
        rf_cm=rf_cm
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        "\nDecision Tree Accuracy:",
        round(dt_accuracy, 4)
    )

    print(
        "Tuned Decision Tree Accuracy:",
        round(tuned_dt_accuracy, 4)
    )

    print(
        "Random Forest Accuracy:",
        round(rf_accuracy, 4)
    )

    print(
        "\nDecision Tree Mean CV Accuracy:",
        round(dt_cv_scores.mean(), 4)
    )

    print(
        "Random Forest Mean CV Accuracy:",
        round(rf_cv_scores.mean(), 4)
    )

    print("\nGenerated output files:")
    print("outputs/decision_tree.png")
    print("outputs/decision_tree_confusion_matrix.png")
    print("outputs/depth_vs_accuracy.png")
    print("outputs/random_forest_confusion_matrix.png")
    print("outputs/feature_importance.png")
    print("outputs/cross_validation.png")
    print("outputs/model_comparison.png")
    print("outputs/model_results.xlsx")

    print("\n" + "=" * 60)
    print("TASK 5 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
