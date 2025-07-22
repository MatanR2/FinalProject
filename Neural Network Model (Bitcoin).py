# Basic Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load the merged data
print("Loading data...")
df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/merged_bitcoin_data.csv")
print(f"Loaded {len(df)} rows of data")

# 2. Display initial information about the dataset
print("\nDataset information:")
print(f"Columns: {df.columns.tolist()}")
print(f"First few rows:")
print(df.head())

# 3. Check for missing values
print("\nChecking for missing values:")
missing_values = df.isna().sum()
print(missing_values[missing_values > 0])  # Only show columns with missing values

# 4. Prepare features (X) and target (y)
print("\nPreparing features and target...")

# Get list of feature columns (exclude Date and target)
feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
print(f"Using these features: {feature_cols}")

# Create X and y from the same filtered dataframe to ensure same length
# First, check if there are any NaN values in the target column
if df['target'].isna().any():
    print(f"Found {df['target'].isna().sum()} NaN values in target column")
    # Filter out rows with NaN in target
    df_filtered = df.dropna(subset=['target'])
    print(f"After filtering: {len(df_filtered)} rows")

    # Create X and y from the SAME filtered dataframe
    X = df_filtered[feature_cols]
    y = df_filtered['target']
else:
    # If no NaN values in target, use the original dataframe
    X = df[feature_cols]
    y = df['target']

# Ensuring that X and y have the same length
print(f"X shape: {X.shape}, y shape: {y.shape}")
if X.shape[0] != y.shape[0]:
    raise ValueError(f"X and y have different lengths: X={X.shape[0]}, y={y.shape[0]}")

# 5. Data Normalization
print("\nNormalizing data...")
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
print("\nSplitting into train and test sets...")
# Use 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False  # Preserving time order
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# 7. Check the distribution of target classes
print("\nTarget distribution:")
print(y.value_counts())

# 8. Model Building - A Basic FeedForward Neural Network
def build_model(input_dim):
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),  # Regularization to prevent overfitting

        # Hidden layers
        Dense(32, activation='relu'),
        Dropout(0.2),

        # Output layer - binary classification (0: down, 1: up)
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # For binary classification
        metrics=['accuracy']
    )

    return model

# 9. Verify if target is binary, convert if needed
unique_values = np.unique(y)
print(f"\nUnique target values: {unique_values}")

if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0.0, 1.0]):
    print("Converting target to binary classification (0: down/neutral, 1: up)")
    # Convert to binary: 1 for price increase, 0 for decrease or no change
    y_binary = (y > 0).astype(int)
    # Update train and test sets
    _, _, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, shuffle=False
    )
    print(f"Binary target distribution: {np.bincount(y_binary.astype(int))}")
    # Update y to binary version for use in optimization
    y = y_binary
else:
    print("Target is already binary, no conversion needed")

# Save preprocessed data as global variables for use in other cells
input_dim = X_scaled.shape[1]  # Number of features

# 10. Model Training
print("\nTraining the baseline model...")
# Create model
model = build_model(input_dim)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Training the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Get the number of epochs actually trained
actual_epochs = len(history.history['loss'])
print(f"Model trained for {actual_epochs} epochs before early stopping")

# 11. Model Evaluation
print("\nEvaluating the baseline model...")
# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 12. Visualize Results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Baseline Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Baseline Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('/content/drive/My Drive/Colab Notebooks/baseline_training_history.png')
plt.show()

# 13. Print model summary
print("\nBaseline Model Summary:")
model.summary()

#Hyperparameters Optimization

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import json
from itertools import product
import time

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Define model building function for hyperparameter tuning with binary classification
def create_model(neurons1=64, neurons2=32, dropout1=0.3, dropout2=0.2,
                learning_rate=0.001, activation='relu'):
    model = Sequential([
        Dense(neurons1, activation=activation, input_dim=input_dim),
        Dropout(dropout1),
        Dense(neurons2, activation=activation),
        Dropout(dropout2),
        Dense(1, activation='sigmoid')  # Since we have a binary classification
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# 2. Comprehensive Grid Search
def perform_grid_search():
    print("Performing comprehensive grid search for hyperparameter tuning...")

    # Define hyperparameter grid
    param_grid = {
        'neurons1': [32, 64, 128],
        'neurons2': [16, 32, 64],
        'dropout1': [0.2, 0.3, 0.4],
        'dropout2': [0.1, 0.2, 0.3],
        'learning_rate': [0.01, 0.001],
        'activation': ['relu', 'tanh']
    }

    # Calculate total number of combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total_combinations = np.prod([len(v) for v in values])

    print(f"Total possible combinations: {total_combinations}")
    print("This could take a while. Consider reducing the parameter space if needed.")

    # Ask if user wants to proceed with all combinations or a random subset
    proceed = input(f"This will test {total_combinations} combinations. Proceed? (yes/no): ")

    if proceed.lower() != 'yes':
        # User chose not to proceed with all combinations
        num_samples = int(input("How many random combinations to try? (suggested: 10-20): "))

        # Generate random indices for sampling
        all_combinations = list(product(*values))
        sample_indices = np.random.choice(len(all_combinations), size=num_samples, replace=False)
        combinations_to_try = [dict(zip(keys, all_combinations[i])) for i in sample_indices]

        print(f"Sampling {num_samples} random combinations from the parameter space")
    else:
        # Generate all combinations
        combinations_to_try = [dict(zip(keys, combo)) for combo in product(*values)]
        print(f"Testing all {total_combinations} combinations")

    # Create a validation set from the training data - 20% of training data
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_sub = X_train[:-val_size]
    y_train_sub = y_train[:-val_size]

    # Track best parameters and performance
    best_val_acc = 0
    best_params = {}
    results = []

    # Start timing
    start_time = time.time()

    # Test each combination
    for i, params in enumerate(combinations_to_try):
        print(f"\nTesting combination {i+1}/{len(combinations_to_try)}: {params}")

        # Create and train model with these parameters
        model = create_model(**params)

        # Early stopping for efficiency
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model with minimal output
        history = model.fit(
            X_train_sub, y_train_sub,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0  # Set to 1 for more detailed output
        )

        # Get validation accuracy from the last epoch
        val_acc = max(history.history['val_accuracy'])

        # Track results
        results.append((params, val_acc))
        print(f"Validation accuracy: {val_acc:.4f}")

        # Update best parameters if needed
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            print(f"New best parameters found!")

        # Show elapsed time and estimated time remaining
        elapsed = time.time() - start_time
        avg_time_per_combo = elapsed / (i + 1)
        remaining_combos = len(combinations_to_try) - (i + 1)
        est_time_remaining = avg_time_per_combo * remaining_combos

        print(f"Elapsed time: {elapsed:.1f}s, Est. time remaining: {est_time_remaining:.1f}s")

    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nGrid search completed in {total_time:.2f} seconds")

    # Sort results by validation accuracy
    results.sort(key=lambda x: x[1], reverse=True)

    # Display top 5 results
    print("\nTop 5 parameter combinations:")
    for i, (params, acc) in enumerate(results[:5]):
        print(f"{i+1}. Accuracy: {acc:.4f}, Parameters: {params}")

    print(f"\nBest parameters: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Visualize hyperparameter impact
    visualize_hyperparameter_impact(results)

    # Save the best parameters
    with open("/content/drive/My Drive/Colab Notebooks/best_params.json", "w") as f:
        json.dump(best_params, f)

    return best_params

# Helper function to visualize hyperparameter impact
def visualize_hyperparameter_impact(results):
    # Extract parameters and accuracies
    params_list = [r[0] for r in results]
    accuracies = [r[1] for r in results]

    # Create figures for each hyperparameter
    for param_name in ['neurons1', 'neurons2', 'dropout1', 'dropout2', 'learning_rate', 'activation']:
        plt.figure(figsize=(10, 6))

        # Group by this parameter
        param_values = [p[param_name] for p in params_list]
        unique_values = sorted(set(param_values))

        # Calculate average accuracy for each value
        avg_accs = []
        for val in unique_values:
            indices = [i for i, p in enumerate(param_values) if p == val]
            avg_acc = np.mean([accuracies[i] for i in indices])
            avg_accs.append(avg_acc)

        # Plot
        plt.bar([str(v) for v in unique_values], avg_accs)
        plt.title(f'Impact of {param_name} on Validation Accuracy')
        plt.xlabel(param_name)
        plt.ylabel('Average Validation Accuracy')
        plt.ylim(min(avg_accs) - 0.05, max(avg_accs) + 0.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'/content/drive/My Drive/Colab Notebooks/{param_name}_impact.png')
        plt.show()

# 3. Batch Size Analysis
def analyze_batch_size(best_params):
    print("Analyzing batch size impact...")

    # Try different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]
    accuracies = []
    histories = []

    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")
        model = create_model(**best_params)

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        # Evaluate model
        results = model.evaluate(X_test, y_test, verbose=0)
        print(f"Batch Size: {batch_size}, Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")
        accuracies.append((batch_size, results[1]))
        histories.append((batch_size, history))

    # Find optimal batch size
    optimal_batch_size = max(accuracies, key=lambda x: x[1])[0]
    print(f"Optimal batch size: {optimal_batch_size}")

    # Plot batch size vs accuracy
    batch_sizes_values = [x[0] for x in accuracies]
    accuracy_values = [x[1] for x in accuracies]

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes_values, accuracy_values, 'o-')
    plt.title('Batch Size vs Test Accuracy')
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/Colab Notebooks/batch_size_analysis.png')
    plt.show()

    # Save optimal batch size
    best_params['optimal_batch_size'] = optimal_batch_size

    return optimal_batch_size

# 4. Network Architecture Analysis
def analyze_network_architecture(best_params=None):
    print("Analyzing network architecture using best hyperparameters from grid search...")

    # Try to load best parameters if not provided
    if best_params is None:
        try:
            with open("/content/drive/My Drive/Colab Notebooks/best_params.json", "r") as f:
                best_params = json.load(f)
            print(f"Loaded best parameters: {best_params}")
        except:
            # Default parameters if file not found
            best_params = {
                'neurons1': 64,
                'neurons2': 32,
                'dropout1': 0.3,
                'dropout2': 0.2,
                'learning_rate': 0.001,
                'activation': 'relu'
            }
            print(f"Using default parameters: {best_params}")

    # Extract optimized parameters
    neurons1 = best_params['neurons1']
    neurons2 = best_params['neurons2']
    dropout1 = best_params['dropout1']
    dropout2 = best_params['dropout2']
    learning_rate = best_params['learning_rate']
    activation = best_params['activation']

    # Define different architectures using the optimized hyperparameters
    architectures = [
        ('Single Layer', [(neurons1, activation, dropout1)]),
        ('Two Layers', [(neurons1, activation, dropout1), (neurons2, activation, dropout2)]),
        ('Three Layers', [(neurons1, activation, dropout1), (neurons2, activation, dropout2),
                          (neurons2 // 2, activation, dropout2 / 2)]),
        ('Wide Network', [(neurons1 * 2, activation, dropout1), (neurons2 * 2, activation, dropout2)]),
        ('Deep Network', [(neurons1, activation, dropout1), (neurons2, activation, dropout2),
                          (neurons2 // 2, activation, dropout2 / 2), (neurons2 // 4, activation, dropout2 / 4)])
    ]

    print(f"Testing architectures with optimized parameters:")
    print(f"  - Activation: {activation}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Base neurons: {neurons1}, {neurons2}")
    print(f"  - Base dropout: {dropout1}, {dropout2}")

    results = []

    for name, layers in architectures:
        print(f"\nTesting architecture: {name}")
        print(f"Layer configuration: {layers}")

        # Create model with this architecture
        model = Sequential()

        # Add input layer
        model.add(Dense(layers[0][0], activation=layers[0][1], input_dim=input_dim))
        model.add(Dropout(layers[0][2]))

        # Add hidden layers
        for neurons, activation, dropout_rate in layers[1:]:
            model.add(Dense(neurons, activation=activation))
            model.add(Dropout(dropout_rate))

        # Add output layer - binary classification
        model.add(Dense(1, activation='sigmoid'))

        # Compile model using the optimized learning rate
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train the model
        batch_size = best_params.get('optimal_batch_size', 32)  # Use optimized batch size if available
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{name}: Test accuracy: {test_acc:.4f}")

        # Store results and model complexity
        num_params = model.count_params()
        results.append((name, test_acc, num_params, len(layers)))

    # Plot architecture comparison
    names = [r[0] for r in results]
    accs = [r[1] for r in results]
    params = [r[2] for r in results]
    num_layers = [r[3] for r in results]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy comparison
    bars = ax1.bar(names, accs, color='skyblue')
    ax1.set_title('Architecture Comparison - Accuracy')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_ylim(min(accs) - 0.05, max(accs) + 0.05)
    ax1.tick_params(axis='x', rotation=45)

    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', fontsize=9)

    # Model complexity comparison (number of parameters)
    bars = ax2.bar(names, params, color='lightgreen')
    ax2.set_title('Architecture Comparison - Complexity')
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Number of Parameters')
    ax2.set_yscale('log')  # Log scale for better visualization

    # Add parameter count on top of bars
    for bar, param in zip(bars, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f'{param:,}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('/content/drive/My Drive/Colab Notebooks/architecture_comparison.png')
    plt.show()

    # Additional plot - accuracy vs complexity
    plt.figure(figsize=(10, 6))
    plt.scatter(params, accs, s=100, alpha=0.7)

    # Add labels to each point
    for name, acc, param in zip(names, accs, params):
        plt.annotate(name, (param, acc),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title('Accuracy vs Model Complexity')
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Test Accuracy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('/content/drive/My Drive/Colab Notebooks/accuracy_vs_complexity.png')
    plt.show()

    # Find best architecture
    best_architecture = max(results, key=lambda x: x[1])
    print(f"\nBest architecture: {best_architecture[0]} with accuracy {best_architecture[1]:.4f}")
    print(f"Number of parameters: {best_architecture[2]:,}")

    return best_architecture[0]

# 5. Feature Importance Analysis using permutation
def analyze_feature_importance():
    print("Analyzing feature importance...")

    # Get the list of feature names
    feature_names = feature_cols

    # Create a baseline model with best parameters
    try:
        with open("/content/drive/My Drive/Colab Notebooks/best_params.json", "r") as f:
            best_params = json.load(f)
    except:
        best_params = {
            'neurons1': 64,
            'neurons2': 32,
            'dropout1': 0.3,
            'dropout2': 0.2,
            'learning_rate': 0.001,
            'activation': 'relu'
        }

    # Train baseline model
    baseline_model = create_model(**best_params)
    baseline_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    baseline_acc = baseline_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Create a model for each feature permutation
    feature_importances = []

    for i in range(len(feature_names)):
        # Create a copy of the test data
        X_test_permuted = X_test.copy()

        # Permute a single feature
        np.random.shuffle(X_test_permuted[:, i])

        # Evaluate with permuted feature
        permuted_acc = baseline_model.evaluate(X_test_permuted, y_test, verbose=0)[1]

        # Calculate importance (decrease in accuracy)
        importance = baseline_acc - permuted_acc
        feature_importances.append((feature_names[i], importance))

    # Sort features by importance
    feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    features = [x[0] for x in feature_importances]
    importances = [x[1] for x in feature_importances]

    colors = ['green' if i >= 0 else 'red' for i in importances]

    plt.barh(features, importances, color=colors)
    plt.title('Feature Importance (Impact on Accuracy)')
    plt.xlabel('Decrease in Accuracy When Feature is Permuted')
    plt.tight_layout()
    plt.savefig('/content/drive/My Drive/Colab Notebooks/feature_importance.png')
    plt.show()

    print("Feature importances (higher value = more important):")
    for feature, importance in feature_importances:
        print(f"{feature}: {importance:.4f}")

    return feature_importances

# Main function to run optimization
def run_hyperparameter_optimization():
    # 1. Grid search for best parameters
    best_params = perform_grid_search()

    # 2. Analyze batch size using best parameters
    optimal_batch_size = analyze_batch_size(best_params)

    # 3. Analyze network architectures using best parameters
    best_architecture = analyze_network_architecture(best_params)

    # 4. Analyze feature importance
    feature_importances = analyze_feature_importance()

    # 5. Save final optimized parameters
    final_params = best_params.copy()
    final_params['optimal_batch_size'] = optimal_batch_size
    final_params['best_architecture'] = best_architecture

    with open("/content/drive/My Drive/Colab Notebooks/optimized_params.json", "w") as f:
        json.dump(final_params, f)

    print("\nOptimization complete!")
    print(f"Final optimized parameters: {final_params}")

    return final_params

# Uncomment to run specific analyses
# best_params = perform_grid_search()
# optimal_batch_size = analyze_batch_size(best_params)
#best_architecture = analyze_network_architecture()
# feature_importances = analyze_feature_importance()

# Uncomment to run the full optimization pipeline
optimized_params = run_hyperparameter_optimization()


# Train and test the optimized neural network
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import datetime

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_optimized_model(architecture_type=None):
    """
    Build a model with the optimized parameters.
    If architecture_type is provided, it will override the saved best architecture.
    """
    try:
        # Load optimized parameters
        with open("/content/drive/My Drive/Colab Notebooks/optimized_params.json", "r") as f:
            params = json.load(f)
        print("Loaded optimized parameters:", params)
    except:
        print("No optimized parameters found, using default values")
        params = {
            'neurons1': 64,
            'neurons2': 32,
            'dropout1': 0.3,
            'dropout2': 0.2,
            'learning_rate': 0.001,
            'activation': 'relu',
            'optimal_batch_size': 32,
            'best_architecture': 'Two Layers'
        }

    # Override architecture if provided
    if architecture_type:
        params['best_architecture'] = architecture_type

    # Build model based on best architecture
    arch = params['best_architecture']
    model = Sequential()

    if arch == 'Single Layer':
        model.add(Dense(params['neurons1'], activation=params['activation'], input_dim=input_dim))
        model.add(Dropout(params['dropout1']))

    elif arch == 'Two Layers':
        model.add(Dense(params['neurons1'], activation=params['activation'], input_dim=input_dim))
        model.add(Dropout(params['dropout1']))
        model.add(Dense(params['neurons2'], activation=params['activation']))
        model.add(Dropout(params['dropout2']))

    elif arch == 'Three Layers':
        model.add(Dense(params['neurons1'], activation=params['activation'], input_dim=input_dim))
        model.add(Dropout(params['dropout1']))
        model.add(Dense(params['neurons2'], activation=params['activation']))
        model.add(Dropout(params['dropout2']))
        model.add(Dense(16, activation=params['activation']))
        model.add(Dropout(0.1))

    elif arch == 'Wide Network':
        model.add(Dense(128, activation=params['activation'], input_dim=input_dim))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation=params['activation']))
        model.add(Dropout(0.3))

    elif arch == 'Deep Network':
        model.add(Dense(64, activation=params['activation'], input_dim=input_dim))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation=params['activation']))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation=params['activation']))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation=params['activation']))
        model.add(Dropout(0.1))

    else:  # Default to Two Layers if architecture not recognized
        print(f"Architecture '{arch}' not recognized, defaulting to Two Layers")
        model.add(Dense(params['neurons1'], activation=params['activation'], input_dim=input_dim))
        model.add(Dropout(params['dropout1']))
        model.add(Dense(params['neurons2'], activation=params['activation']))
        model.add(Dropout(params['dropout2']))

    # Add output layer - binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, params['optimal_batch_size']

def train_and_evaluate_final_model():
    """Train the final model with the optimized parameters and evaluate it"""
    print("Training final model with optimized parameters...")

    # Build optimized model
    model, batch_size = build_optimized_model()

    # Set up callbacks
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # Checkpoint to save best model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"/content/drive/My Drive/Colab Notebooks/bitcoin_model_checkpoint_{timestamp}.h5"
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # Train model with optimized parameters
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Use more epochs with early stopping
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")

    # Generate predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the test predictions with dates
    import pandas as pd
    test_predictions_nn = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'y_pred_proba': y_pred_proba.flatten()
    })
    test_predictions_nn.to_csv("/content/drive/My Drive/Colab Notebooks/nn_test_predictions.csv")
    print("NN test predictions saved!")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/content/drive/My Drive/Colab Notebooks/confusion_matrix.png')
    plt.show()

    # Learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('/content/drive/My Drive/Colab Notebooks/learning_curves.png')
    plt.show()

    # Save the final model
    model_path = f"/content/drive/My Drive/Colab Notebooks/bitcoin_price_prediction_model_{timestamp}.h5"
    save_model(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save with consistent naming for the trading strategy
    model.save("/content/drive/My Drive/Colab Notebooks/final_nn_model.h5")
    print("Neural Network model saved as 'final_nn_model.h5'")

    # Save the scaler used for preprocessing
    import joblib
    # The scaler should be available from the global scope or recreate it
    # If you have access to the scaler used earlier:
    if 'scaler' in globals():
        joblib.dump(scaler, "/content/drive/My Drive/Colab Notebooks/nn_scaler.pkl")
        print("Neural Network scaler saved as 'nn_scaler.pkl'")
    else:
        # Recreate the scaler (this should match your original preprocessing)
        from sklearn.preprocessing import StandardScaler
        temp_scaler = StandardScaler()
        temp_scaler.fit(X_train)  # X_train should be available in scope
        joblib.dump(temp_scaler, "/content/drive/My Drive/Colab Notebooks/nn_scaler.pkl")
        print("Neural Network scaler recreated and saved as 'nn_scaler.pkl'")

    # Return metrics for future reference
    return {
        'test_accuracy': test_acc,
        'model_path': model_path,
        'final_model_path': "/content/drive/My Drive/Colab Notebooks/final_nn_model.h5"

    }

# Run the final model training and evaluation
train_and_evaluate_final_model()


# Debug Neural Network Performance Issues
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=== STEP 1: בדיקת חלוקת הנתונים ===")

# Load data (replace with your actual path)
df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/merged_bitcoin_data.csv")

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
if df['target'].isna().any():
    df_filtered = df.dropna(subset=['target'])
    X = df_filtered[feature_cols]
    y = df_filtered['target']
else:
    X = df[feature_cols]
    y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check target distribution
print(f"Total samples: {len(X_scaled)}")
print(f"Unique target values: {np.unique(y)}")

# Convert target to binary if needed (handle -1, 0, 1 case)
if np.min(y) < 0:
    print("Converting target from {-1, 0, 1} to {0, 1}")
    # Convert: -1 and 0 -> 0 (down/no change), 1 -> 1 (up)
    y_binary = (y > 0).astype(int)
    print(f"Original target distribution: {np.unique(y, return_counts=True)}")
    print(f"Binary target distribution: {np.bincount(y_binary)}")
    y = y_binary
else:
    print(f"Target distribution: {np.bincount(y.astype(int))}")

# Create FIXED train/test split that we'll use consistently
X_train_fixed, X_test_fixed, y_train_fixed, y_test_fixed = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Fixed train size: {len(X_train_fixed)}")
print(f"Fixed test size: {len(X_test_fixed)}")
print(f"Train target distribution: {np.bincount(y_train_fixed.astype(int))}")
print(f"Test target distribution: {np.bincount(y_test_fixed.astype(int))}")

print("\n=== STEP 2: מודל בסיסי עם הגדרות קבועות ===")

def create_baseline_model(input_dim):
    """Create baseline model with default parameters"""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Train baseline model
print("Training baseline model...")
baseline_model = create_baseline_model(X_scaled.shape[1])

# Use FIXED validation split manually
val_size = int(0.2 * len(X_train_fixed))
X_train_base = X_train_fixed[:-val_size]
X_val_base = X_train_fixed[-val_size:]
y_train_base = y_train_fixed[:-val_size]
y_val_base = y_train_fixed[-val_size:]

print(f"Baseline train size: {len(X_train_base)}")
print(f"Baseline val size: {len(X_val_base)}")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

baseline_history = baseline_model.fit(
    X_train_base, y_train_base,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_base, y_val_base),  # Fixed validation data
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate baseline on fixed test set
baseline_test_loss, baseline_test_acc = baseline_model.evaluate(X_test_fixed, y_test_fixed, verbose=0)
print(f"Baseline test accuracy: {baseline_test_acc:.4f}")

print("\n=== STEP 3: בדיקת אילו פרמטרים נבדקים ב-Grid Search ===")

# Define the exact same grid as in your code
param_grid = {
    'neurons1': [32, 64, 128],
    'neurons2': [16, 32, 64],
    'dropout1': [0.2, 0.3, 0.4],
    'dropout2': [0.1, 0.2, 0.3],
    'learning_rate': [0.01, 0.001],
    'activation': ['relu', 'tanh']
}

# Check if default parameters are in the grid
default_params = {
    'neurons1': 64,
    'neurons2': 32,
    'dropout1': 0.3,
    'dropout2': 0.2,
    'learning_rate': 0.001,
    'activation': 'relu'
}

print("Default parameters:")
for key, value in default_params.items():
    in_grid = value in param_grid[key]
    print(f"  {key}: {value} - {'✓ IN GRID' if in_grid else '✗ NOT IN GRID'}")

print("\n=== STEP 4: Grid Search עם אותה חלוקת נתונים ===")

def create_model_with_params(input_dim, **params):
    """Create model with specific parameters"""
    model = Sequential([
        Dense(params['neurons1'], activation=params['activation'], input_dim=input_dim),
        Dropout(params['dropout1']),
        Dense(params['neurons2'], activation=params['activation']),
        Dropout(params['dropout2']),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Test the DEFAULT parameters specifically
print("Testing DEFAULT parameters in grid search conditions...")

# Use the SAME train/val split as baseline
default_model = create_model_with_params(X_scaled.shape[1], **default_params)

# Use same early stopping as grid search
grid_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Same as grid search
    restore_best_weights=True
)

default_grid_history = default_model.fit(
    X_train_base, y_train_base,  # Same data as baseline
    epochs=20,  # Same as grid search
    batch_size=32,
    validation_data=(X_val_base, y_val_base),
    callbacks=[grid_early_stopping],
    verbose=1
)

# Evaluate on same test set
default_grid_test_loss, default_grid_test_acc = default_model.evaluate(X_test_fixed, y_test_fixed, verbose=0)
print(f"Default params in grid conditions test accuracy: {default_grid_test_acc:.4f}")

print("\n=== STEP 5: השוואת תוצאות ===")

print("Comparison:")
print(f"Baseline (50 epochs, patience=10): {baseline_test_acc:.4f}")
print(f"Default in grid conditions (20 epochs, patience=5): {default_grid_test_acc:.4f}")
print(f"Difference: {baseline_test_acc - default_grid_test_acc:.4f}")

if abs(baseline_test_acc - default_grid_test_acc) > 0.01:
    print("\n🔍 MAIN ISSUES IDENTIFIED:")
    if baseline_test_acc > default_grid_test_acc:
        print("- Grid search uses too few epochs (20 vs 50)")
        print("- Grid search uses more aggressive early stopping (patience=5 vs 10)")
    print("- This explains why grid search finds 'worse' parameters")

print("\n=== STEP 6: בדיקת המודל הסופי ===")

# Try to load optimized parameters
try:
    with open("/content/drive/My Drive/Colab Notebooks/optimized_params.json", "r") as f:
        optimized_params = json.load(f)
    print(f"Loaded optimized parameters: {optimized_params}")

    # Extract relevant parameters for model creation
    model_params = {
        'neurons1': optimized_params['neurons1'],
        'neurons2': optimized_params['neurons2'],
        'dropout1': optimized_params['dropout1'],
        'dropout2': optimized_params['dropout2'],
        'learning_rate': optimized_params['learning_rate'],
        'activation': optimized_params['activation']
    }

    print("Testing optimized parameters with PROPER training conditions...")

    optimized_model = create_model_with_params(X_scaled.shape[1], **model_params)

    # Use same conditions as baseline (proper training)
    optimized_history = optimized_model.fit(
        X_train_base, y_train_base,
        epochs=50,  # Proper number of epochs
        batch_size=32,
        validation_data=(X_val_base, y_val_base),
        callbacks=[early_stopping],  # Proper early stopping
        verbose=1
    )

    optimized_test_loss, optimized_test_acc = optimized_model.evaluate(X_test_fixed, y_test_fixed, verbose=0)
    print(f"Optimized params with proper training: {optimized_test_acc:.4f}")

except FileNotFoundError:
    print("No optimized parameters file found - run grid search first")
    optimized_test_acc = None

print("\n=== FINAL COMPARISON ===")
print(f"Baseline (proper training):     {baseline_test_acc:.4f}")
if optimized_test_acc:
    print(f"Optimized (proper training):    {optimized_test_acc:.4f}")
    print(f"Improvement:                    {optimized_test_acc - baseline_test_acc:.4f}")

print("\n=== RECOMMENDATIONS ===")
print("1. Use same number of epochs in grid search as in final training")
print("2. Use same early stopping patience")
print("3. Use same train/val/test splits throughout")
print("4. Set random seeds properly")
print("5. Consider validation_split vs validation_data consistency")
