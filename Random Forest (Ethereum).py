import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import time
import joblib
from scipy.stats import randint, uniform

# Verify the target distribution
print("\nTarget distribution:")
print(y.value_counts())

# Convert to binary classification if needed (1 for up, 0 for down/unchanged)
if not np.array_equal(np.unique(y), [0, 1]):
    print("Converting target to binary classification (0: down/neutral, 1: up)")
    y_binary = (y > 0).astype(int)
    print(f"Binary target distribution: {np.bincount(y_binary)}")
    y = y_binary


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train a default Random Forest model and evaluate it"""
    print("\n--- Random Forest with Default Parameters ---")

    # Create and train model
    start_time = time.time()
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Random Forest - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('/content/drive/My Drive/Colab Notebooks/rf_confusion_matrix.png')
    plt.show()

    # Plot feature importance
    plot_feature_importances(rf_model, feature_names, "Random Forest", "rf")

    return rf_model, accuracy, y_pred, y_pred_proba

def plot_feature_importances(model, feature_names, model_name, prefix):
    """Plot feature importances for a given model"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances from model
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Plot importances
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} - Feature Importances')
        plt.bar(range(len(indices)), importances[indices], color='r', yerr=std[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(indices)])
        plt.tight_layout()
        plt.savefig(f'/content/drive/My Drive/Colab Notebooks/{prefix}_feature_importance.png')
        plt.show()

        # Print top features
        print("\nTop 5 features:")
        for i in range(5):
            if i < len(indices):
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def hyperparameter_tuning_random_forest(X_train, y_train, cv=5):
    """Perform hyperparameter tuning for Random Forest"""
    print("\n--- Random Forest Hyperparameter Tuning ---")

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced']
    }

    # Calculate total combinations
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Total parameter combinations: {total_combinations}")

    print(f"Using full GridSearchCV with all {total_combinations} combinations")
    search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

    # Perform the search
    start_time = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start_time

    print(f"\nHyperparameter tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation score: {search.best_score_:.4f}")

    # Get the best estimator
    best_rf = search.best_estimator_

    # Save the model
    joblib.dump(best_rf, '/content/drive/My Drive/Colab Notebooks/best_rf_model.pkl')

    return best_rf, search.best_params_, search.best_score_

def evaluate_tuned_model(model, X_test, y_test, feature_names, model_name="Tuned Random Forest", prefix="tuned_rf"):
    """Evaluate a tuned model"""
    print(f"\n--- Evaluating {model_name} ---")

    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Prediction time: {prediction_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'/content/drive/My Drive/Colab Notebooks/{prefix}_confusion_matrix.png')
    plt.show()

    # Plot feature importance
    plot_feature_importances(model, feature_names, model_name, prefix)

    return accuracy, y_pred, y_pred_proba

def compare_models(model1, model2, X_test, y_test, model1_name="Default RF", model2_name="Tuned RF"):
    """Compare two models' predictions and show where they differ"""
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)

    # Calculate agreement percentage
    agreement = np.mean(y_pred1 == y_pred2) * 100
    print(f"\nModel agreement: {agreement:.2f}%")

    # Where models differ
    diff_indices = np.where(y_pred1 != y_pred2)[0]
    if len(diff_indices) > 0:
        print(f"\nThe models disagree on {len(diff_indices)} samples out of {len(y_test)} ({len(diff_indices)/len(y_test)*100:.2f}%)")

        # See which model is correct in those cases
        # Access directly using integer indices to avoid the pandas KeyError
        y_test_array = np.array(y_test) if hasattr(y_test, '__array__') else y_test

        model1_correct = sum(y_pred1[diff_indices] == y_test_array[diff_indices])
        model2_correct = sum(y_pred2[diff_indices] == y_test_array[diff_indices])

        print(f"{model1_name} is correct on {model1_correct} disagreed samples ({model1_correct/len(diff_indices)*100:.2f}%)")
        print(f"{model2_name} is correct on {model2_correct} disagreed samples ({model2_correct/len(diff_indices)*100:.2f}%)")
    else:
        print("Both models make identical predictions.")


def run_random_forest_pipeline(use_all_combinations=True):
    """Run the full Random Forest model optimization pipeline

    Parameters:
    -----------
    use_all_combinations : bool, default=True
        If True, use GridSearchCV to test all parameter combinations.
        If False, use RandomizedSearchCV to test only 50 combinations.
    """
    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Get feature names
    feature_names = feature_cols

    # 2. Train and evaluate baseline model
    rf_model, baseline_accuracy, baseline_pred, baseline_proba = train_and_evaluate_random_forest(
        X_train, X_test, y_train, y_test, feature_names
    )

    # 3. Perform hyperparameter tuning
    best_rf, best_params, best_cv_score = hyperparameter_tuning_random_forest(
        X_train, y_train)

    # 4. Evaluate tuned model
    tuned_accuracy, tuned_pred, tuned_proba = evaluate_tuned_model(
        best_rf, X_test, y_test, feature_names, "Tuned Random Forest", "tuned_rf"
    )

    # 5. Compare baseline and tuned models
    compare_models(rf_model, best_rf, X_test, y_test, "Baseline RF", "Tuned RF")

    # Save the test predictions
    import pandas as pd
    test_predictions_rf = pd.DataFrame({
    'y_true': y_test,
    'y_pred': tuned_pred
    })
    test_predictions_rf.to_csv("/content/drive/My Drive/Colab Notebooks/rf_test_predictions.csv")
    print("RF test predictions saved!")

    print("\nRandom Forest pipeline completed successfully!")

    # Save the models
    import joblib
    joblib.dump(best_rf, "/content/drive/My Drive/Colab Notebooks/final_rf_model.pkl")
    print("Random Forest model saved as 'final_rf_model.pkl'")

    # Save the scaler (X_scaled was created from scaler.fit_transform)
    # You need to recreate the scaler since it wasn't saved earlier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on training data
    joblib.dump(scaler, "/content/drive/My Drive/Colab Notebooks/rf_scaler.pkl")
    print("Random Forest scaler saved as 'rf_scaler.pkl'")

    return {
        'baseline_model': rf_model,
        'tuned_model': best_rf,
        'baseline_accuracy': baseline_accuracy,
        'tuned_accuracy': tuned_accuracy,
        'best_params': best_params
    }

# To run with all combinations:
results = run_random_forest_pipeline()

# To run with only 50 random combinations:
# results = run_random_forest_pipeline(use_all_combinations=False)
