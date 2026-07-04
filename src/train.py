"""
Model Training Pipeline
-----------------------
This script orchestrates the complete machine learning workflow:

1. Load dataset
2. Preprocess data
3. Train SVM
4. Train Decision Tree
5. Evaluate models
6. Save trained models
7. Save comparison results
"""

from src.logger import logger

from src.data_preprocessing import (
    load_data,
    split_features_target,
    scale_features,
    split_data
)

from src.svm_model import (
    train_svm,
    predict_svm,
    save_svm_model
)

from src.decision_tree import (
    train_decision_tree,
    predict_decision_tree,
    save_decision_tree
)

from src.evaluation import (
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    compare_models,
    save_comparison
)


def main():

    logger.info("Loading Dataset...")

    df = load_data()

    X, y = split_features_target(df)

    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = split_data(
        X_scaled,
        y
    )

    logger.info("Dataset Loaded Successfully!")

    # -------------------------
    # Train SVM
    # -------------------------

    logger.info("Training Support Vector Machine...")

    svm_model = train_svm(
        X_train,
        y_train
    )

    svm_predictions = predict_svm(
        svm_model,
        X_test
    )

    save_svm_model(svm_model)

    logger.info("SVM Model Saved Successfully!")

    svm_metrics = evaluate_model(
        y_test,
        svm_predictions
    )

    print("\n" + "=" * 60)
    print("SVM Classification Report")
    print("=" * 60)

    print_classification_report(
        y_test,
        svm_predictions
    )

    plot_confusion_matrix(
        y_test,
        svm_predictions,
        "SVM Confusion Matrix"
    )

    # -------------------------
    # Train Decision Tree
    # -------------------------

    logger.info("Training Decision Tree...")

    dt_model = train_decision_tree(
        X_train,
        y_train
    )

    dt_predictions = predict_decision_tree(
        dt_model,
        X_test
    )

    save_decision_tree(dt_model)

    logger.info("Decision Tree Model Saved Successfully!")

    dt_metrics = evaluate_model(
        y_test,
        dt_predictions
    )

    print("\n" + "=" * 60)
    print("Decision Tree Classification Report")
    print("=" * 60)

    print_classification_report(
        y_test,
        dt_predictions
    )

    plot_confusion_matrix(
        y_test,
        dt_predictions,
        "Decision Tree Confusion Matrix"
    )

    # -------------------------
    # Compare Models
    # -------------------------

    comparison = compare_models(
        svm_metrics,
        dt_metrics
    )

    save_comparison(comparison)

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(comparison)

    logger.info("Model Comparison Saved Successfully!")
    logger.info("Training Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()