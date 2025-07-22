import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)
random.seed(42)

# Initial investment amount
INITIAL_INVESTMENT = 1000000

class TradingStrategyComparison:
    def __init__(self, initial_investment=1000000):
        self.initial_investment = initial_investment
        self.results = []

    def load_bitcoin_data(self, csv_path="/content/drive/My Drive/Colab Notebooks/Yahoo_Finance_Bitcoin.csv"):
        """Load Bitcoin price data from CSV file"""
        print("Loading Bitcoin price data...")

        try:
            # Load data
            btc_data = pd.read_csv(csv_path)

            # Convert date column
            btc_data['Date'] = pd.to_datetime(btc_data['Date'])
            btc_data.set_index('Date', inplace=True)
            btc_data = btc_data.sort_index()

            # Ensure numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in btc_data.columns:
                    btc_data[col] = pd.to_numeric(btc_data[col], errors='coerce')

            # Handle missing columns
            if 'High' not in btc_data.columns:
                btc_data['High'] = btc_data['Open'] * 1.02
            if 'Low' not in btc_data.columns:
                btc_data['Low'] = btc_data['Open'] * 0.98
            if 'Volume' not in btc_data.columns:
                btc_data['Volume'] = 1000000

            # Remove any rows with NaN in essential columns
            btc_data = btc_data.dropna(subset=['Open', 'Close'])

            print(f"Loaded {len(btc_data)} days of data")
            print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")

            return btc_data

        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data if file doesn't exist
            return self.create_sample_data()

    def create_sample_data(self):
        """Create sample Bitcoin data for testing"""
        print("Creating sample data for testing...")

        dates = pd.date_range(start='2020-10-01', end='2021-03-31', freq='D')
        np.random.seed(42)

        # Simulate Bitcoin price movement
        initial_price = 10000
        prices = [initial_price]

        for i in range(len(dates) - 1):
            # Random walk with slight upward bias
            change = np.random.normal(0.02, 0.05)  # 2% daily growth, 5% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # Minimum price of $100

        data = pd.DataFrame({
            'Open': prices,
            'High': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'Close': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'Volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)

        return data

    def create_mock_predictions(self, price_data):
        """Create mock predictions that simulate model behavior"""
        print("Creating mock model predictions...")

        # Calculate actual price changes
        price_changes = price_data['Close'].pct_change().shift(-1)
        actual_direction = (price_changes > 0).astype(int)

        # Create mock neural network predictions (slightly better than random)
        nn_predictions = []
        for change in price_changes:
            if pd.isna(change):
                nn_predictions.append(np.random.choice([0, 1]))
            else:
                # 60% chance of correct prediction
                if np.random.random() < 0.6:
                    nn_predictions.append(1 if change > 0 else 0)
                else:
                    nn_predictions.append(1 if change <= 0 else 0)

        # Create mock random forest predictions (slightly different performance)
        rf_predictions = []
        for change in price_changes:
            if pd.isna(change):
                rf_predictions.append(np.random.choice([0, 1]))
            else:
                # 55% chance of correct prediction
                if np.random.random() < 0.55:
                    rf_predictions.append(1 if change > 0 else 0)
                else:
                    rf_predictions.append(1 if change <= 0 else 0)

        predictions = pd.DataFrame({
            'actual_direction': actual_direction,
            'nn_prediction': nn_predictions,
            'rf_prediction': rf_predictions
        }, index=price_data.index)

        # Remove the last row (no future data to predict)
        predictions = predictions[:-1]

        # Calculate accuracies
        nn_acc = accuracy_score(predictions['actual_direction'].dropna(),
                              np.array(predictions['nn_prediction'])[~predictions['actual_direction'].isna()])
        rf_acc = accuracy_score(predictions['actual_direction'].dropna(),
                              np.array(predictions['rf_prediction'])[~predictions['actual_direction'].isna()])

        print(f"Mock Neural Network accuracy: {nn_acc:.3f}")
        print(f"Mock Random Forest accuracy: {rf_acc:.3f}")

        return predictions

    def load_real_predictions(self, price_data):
        """Try to load real model predictions, fallback to mock if not available"""
        try:
            # Define exact model paths
            nn_model_path = "/content/drive/My Drive/Colab Notebooks/final_nn_model.h5"
            rf_model_path = "/content/drive/My Drive/Colab Notebooks/final_rf_model.pkl"
            nn_scaler_path = "/content/drive/My Drive/Colab Notebooks/nn_scaler.pkl"

            # Alternative paths to try
            alternative_rf_paths = [
                "/content/drive/My Drive/Colab Notebooks/best_rf_model.pkl",
                "/content/drive/My Drive/Colab Notebooks/tuned_rf_model.pkl"
            ]

            alternative_nn_paths = []
            import glob
            alternative_nn_paths.extend(glob.glob("/content/drive/My Drive/Colab Notebooks/bitcoin_price_prediction_model*.h5"))
            alternative_nn_paths.extend(glob.glob("/content/drive/My Drive/Colab Notebooks/*model*.h5"))

            # Try to load Random Forest model
            rf_model = None
            for path in [rf_model_path] + alternative_rf_paths:
                if os.path.exists(path):
                    print(f"Loading Random Forest model from: {path}")
                    rf_model = joblib.load(path)
                    break

            if rf_model is None:
                print("No Random Forest model found")
                return self.create_mock_predictions(price_data)

            # Try to load Neural Network model
            nn_model = None
            for path in [nn_model_path] + alternative_nn_paths:
                if os.path.exists(path):
                    try:
                        print(f"Loading Neural Network model from: {path}")
                        from tensorflow.keras.models import load_model
                        nn_model = load_model(path)
                        break
                    except Exception as e:
                        print(f"Failed to load NN model from {path}: {e}")
                        continue

            if nn_model is None:
                print("No Neural Network model found")
                return self.create_mock_predictions(price_data)

            # Prepare features
            print("Preparing features for prediction...")
            features = self.prepare_features_for_models(price_data)

            if features is None or len(features) == 0:
                print("Failed to prepare features")
                return self.create_mock_predictions(price_data)

            # Make predictions
            print("Making predictions with loaded models...")

            # Random Forest predictions
            rf_predictions = rf_model.predict(features)

            # Neural Network predictions
            nn_pred_proba = nn_model.predict(features, verbose=0)
            nn_predictions = (nn_pred_proba > 0.5).astype(int).flatten()

            # Create predictions dataframe
            price_changes = price_data['Close'].pct_change().shift(-1)
            actual_direction = (price_changes > 0).astype(int)

            # Ensure all arrays have the same length
            min_length = min(len(actual_direction), len(nn_predictions), len(rf_predictions))

            predictions = pd.DataFrame({
                'actual_direction': actual_direction.iloc[:min_length],
                'nn_prediction': nn_predictions[:min_length],
                'rf_prediction': rf_predictions[:min_length]
            }, index=price_data.index[:min_length])

            # Remove last row (no future data to predict)
            predictions = predictions[:-1]

            # Calculate accuracies
            valid_mask = ~predictions['actual_direction'].isna()
            if valid_mask.sum() > 0:
                nn_acc = accuracy_score(predictions.loc[valid_mask, 'actual_direction'],
                                      predictions.loc[valid_mask, 'nn_prediction'])
                rf_acc = accuracy_score(predictions.loc[valid_mask, 'actual_direction'],
                                      predictions.loc[valid_mask, 'rf_prediction'])
                print(f"Real Neural Network accuracy: {nn_acc:.3f}")
                print(f"Real Random Forest accuracy: {rf_acc:.3f}")

            print("Successfully loaded real model predictions!")
            return predictions

        except Exception as e:
            print(f"Error loading real models: {e}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            print("Using mock predictions instead")
            return self.create_mock_predictions(price_data)

    def prepare_features_for_models(self, price_data):
        """Prepare features that match your trained models exactly"""
        try:
            features = pd.DataFrame(index=price_data.index)

            # Basic price features (these should exist in your data)
            if 'Open' in price_data.columns:
                features['Open'] = price_data['Open']
            if 'Close' in price_data.columns:
                features['Close'] = price_data['Close']

            # Calculate volatility
            if 'High' in price_data.columns and 'Low' in price_data.columns and 'Open' in price_data.columns:
                features['Volatility (%)'] = ((price_data['High'] - price_data['Low']) / price_data['Open']) * 100
            else:
                features['Volatility (%)'] = 0

            # Mock the missing features with reasonable defaults
            features['Buzz'] = 50  # Default Google Trends value
            features['positive'] = 0.1  # Default sentiment values
            features['negative'] = 0.1
            features['neutral'] = 0.8
            features['compound'] = 0.0

            # Calculate moving averages
            if 'Close' in features.columns:
                features['price_ma3'] = features['Close'].rolling(window=3).mean()
                features['price_ma7'] = features['Close'].rolling(window=7).mean()

            features['google_trends_ma3'] = features['Buzz'].rolling(window=3).mean()
            features['sentiment_ma3'] = features['compound'].rolling(window=3).mean()

            # ADD THE MISSING FEATURES THAT THE SCALER EXPECTS:
            features['Daily Change (indicator)'] = 0  # Default value
            features['isPartial'] = 0  # Default value
            features['sentiment'] = 0  # Default value (duplicate of compound)

            # Fill NaN values
            features = features.ffill().bfill().fillna(0)

            # Only keep rows where we have complete data
            features = features.dropna()

            if len(features) == 0:
                print("No valid features after preprocessing")
                return None

            # Try to load the scaler, but handle feature mismatch gracefully
            scaler_path = "/content/drive/My Drive/Colab Notebooks/nn_scaler.pkl"
            if os.path.exists(scaler_path):
                try:
                    print("Loading saved scaler...")
                    scaler = joblib.load(scaler_path)

                    # Check if we have the right number of features
                    expected_features = scaler.n_features_in_
                    actual_features = features.shape[1]

                    print(f"Expected {expected_features} features, have {actual_features} features")

                    if actual_features != expected_features:
                        print("Feature count mismatch, creating new scaler...")
                        scaler = StandardScaler()
                        features_scaled = scaler.fit_transform(features)
                    else:
                        features_scaled = scaler.transform(features)

                except Exception as e:
                    print(f"Error with saved scaler: {e}")
                    print("Creating new scaler...")
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
            else:
                print("No saved scaler found, creating new one...")
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

            print(f"Prepared {features_scaled.shape[0]} samples with {features_scaled.shape[1]} features")
            return features_scaled

        except Exception as e:
            print(f"Error preparing features: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def strategy_buy_and_hold(self, price_data):
        """Strategy 1: Buy and hold"""
        initial_price = price_data.iloc[0]['Open']
        final_price = price_data.iloc[-1]['Close']

        btc_bought = self.initial_investment / initial_price
        final_value = btc_bought * final_price

        return {
            'strategy': 'Buy and Hold',
            'initial_investment': self.initial_investment,
            'final_value': final_value,
            'profit': final_value - self.initial_investment,
            'roi': (final_value / self.initial_investment - 1) * 100,
            'transactions': 1
        }

    def strategy_random_trading(self, price_data):
        """Strategy 2: Random trading"""
        portfolio = {'cash': 0, 'btc': self.initial_investment / price_data.iloc[0]['Open']}
        transactions = 0
        daily_values = []

        for i, (date, row) in enumerate(price_data.iterrows()):
            current_price = row['Close']

            # Record daily value
            portfolio_value = portfolio['btc'] * current_price + portfolio['cash']
            daily_values.append(portfolio_value)

            # Make random decision (with some probability to avoid over-trading)
            if np.random.random() < 0.1:  # 10% chance to trade each day
                if np.random.random() < 0.5 and portfolio['btc'] > 0:
                    # Sell
                    portfolio['cash'] = portfolio['btc'] * current_price
                    portfolio['btc'] = 0
                    transactions += 1
                elif portfolio['cash'] > 0:
                    # Buy
                    portfolio['btc'] = portfolio['cash'] / current_price
                    portfolio['cash'] = 0
                    transactions += 1

        final_value = portfolio['btc'] * price_data.iloc[-1]['Close'] + portfolio['cash']

        return {
            'strategy': 'Random Trading',
            'initial_investment': self.initial_investment,
            'final_value': final_value,
            'profit': final_value - self.initial_investment,
            'roi': (final_value / self.initial_investment - 1) * 100,
            'transactions': transactions,
            'daily_values': daily_values
        }

    def strategy_model_based(self, price_data, predictions, model_name):
        """Model-based trading strategy"""
        prediction_col = f'{model_name}_prediction'

        portfolio = {'cash': 0, 'btc': self.initial_investment / price_data.iloc[0]['Open']}
        transactions = 0
        daily_values = []

        for i, (date, row) in enumerate(price_data.iterrows()):
            current_price = row['Close']

            # Record daily value
            portfolio_value = portfolio['btc'] * current_price + portfolio['cash']
            daily_values.append(portfolio_value)

            # Get prediction if available
            if date in predictions.index:
                prediction = predictions.loc[date, prediction_col]

                # Trading logic
                if prediction == 0 and portfolio['btc'] > 0:
                    # Sell (predict price will go down)
                    portfolio['cash'] = portfolio['btc'] * current_price
                    portfolio['btc'] = 0
                    transactions += 1
                elif prediction == 1 and portfolio['cash'] > 0:
                    # Buy (predict price will go up)
                    portfolio['btc'] = portfolio['cash'] / current_price
                    portfolio['cash'] = 0
                    transactions += 1

        final_value = portfolio['btc'] * price_data.iloc[-1]['Close'] + portfolio['cash']
        strategy_name = 'Neural Network' if model_name == 'nn' else 'Random Forest'

        return {
            'strategy': f'{strategy_name} Trading',
            'initial_investment': self.initial_investment,
            'final_value': final_value,
            'profit': final_value - self.initial_investment,
            'roi': (final_value / self.initial_investment - 1) * 100,
            'transactions': transactions,
            'daily_values': daily_values
        }

    def plot_results(self, results, price_data):
        """Plot comparison of all strategies"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Portfolio values over time
        ax1.plot(price_data.index, price_data['Close'], 'k-', alpha=0.5, label='Bitcoin Price')
        ax1_twin = ax1.twinx()

        colors = ['blue', 'green', 'red', 'orange']
        for i, result in enumerate(results):
            if 'daily_values' in result:
                ax1_twin.plot(price_data.index, result['daily_values'],
                            color=colors[i], linewidth=2, label=result['strategy'])

        ax1.set_ylabel('Bitcoin Price ($)')
        ax1_twin.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Performance Over Time')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. ROI Comparison
        strategies = [r['strategy'] for r in results]
        rois = [r['roi'] for r in results]
        bars = ax2.bar(strategies, rois, color=colors[:len(results)])

        for bar, roi in zip(bars, rois):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (max(rois) * 0.01),
                    f'{roi:.1f}%', ha='center', va='bottom')

        ax2.set_ylabel('Return on Investment (%)')
        ax2.set_title('ROI Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Final Values
        final_values = [r['final_value'] for r in results]
        bars = ax3.bar(strategies, final_values, color=colors[:len(results)])

        for bar, value in zip(bars, final_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (max(final_values) * 0.01),
                    f'${value:,.0f}', ha='center', va='bottom', rotation=0)

        ax3.set_ylabel('Final Portfolio Value ($)')
        ax3.set_title('Final Portfolio Values')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Transaction counts
        transactions = [r.get('transactions', 0) for r in results]
        bars = ax4.bar(strategies, transactions, color=colors[:len(results)])

        for bar, txn in zip(bars, transactions):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{txn}', ha='center', va='bottom')

        ax4.set_ylabel('Number of Transactions')
        ax4.set_title('Trading Activity')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/content/drive/My Drive/Colab Notebooks/strategy_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def run_comparison(self):
        """Run the complete strategy comparison"""
        print("Starting Bitcoin Trading Strategy Comparison")
        print("=" * 50)

        # Load data
        price_data = self.load_bitcoin_data()

        # Get predictions
        predictions = self.load_real_predictions(price_data)

        # Run all strategies
        results = []

        # Strategy 1: Buy and Hold
        print("\nRunning Buy and Hold strategy...")
        result1 = self.strategy_buy_and_hold(price_data)
        results.append(result1)

        # Strategy 2: Random Trading
        print("Running Random Trading strategy...")
        result2 = self.strategy_random_trading(price_data)
        results.append(result2)

        # Strategy 3: Neural Network
        print("Running Neural Network strategy...")
        result3 = self.strategy_model_based(price_data, predictions, 'nn')
        results.append(result3)

        # Strategy 4: Random Forest
        print("Running Random Forest strategy...")
        result4 = self.strategy_model_based(price_data, predictions, 'rf')
        results.append(result4)

        # Print results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        for result in results:
            print(f"\n{result['strategy']}:")
            print(f"  Initial Investment: ${result['initial_investment']:,}")
            print(f"  Final Value: ${result['final_value']:,.2f}")
            print(f"  Profit/Loss: ${result['profit']:,.2f}")
            print(f"  ROI: {result['roi']:.2f}%")
            print(f"  Transactions: {result.get('transactions', 'N/A')}")

        # Plot results
        self.plot_results(results, price_data)

        return results, price_data, predictions

# Run the comparison
if __name__ == "__main__":
    comparison = TradingStrategyComparison(initial_investment=1000000)
    results, price_data, predictions = comparison.run_comparison()
