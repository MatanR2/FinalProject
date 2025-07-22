import pandas as pd
import numpy as np
import datetime

# Function to parse dates with support for multiple formats
def parse_date(date_value):
    # Handle NaN/None values
    if pd.isna(date_value):
        return None

    # Check if the value is already a datetime object
    if isinstance(date_value, datetime.datetime):
        return date_value

    # If it's a float or int, convert to string first
    if isinstance(date_value, (float, int)):
        # If it's NaN, return None
        if np.isnan(date_value):
            return None
        # Convert to string
        date_value = str(date_value)

    # Try different date formats
    date_formats = [
        '%d/%m/%Y',    # Day/Month/Year
        '%Y-%m-%d',    # ISO format (Year-Month-Day)
        '%m/%d/%Y',    # Month/Day/Year
        '%Y/%m/%d'     # Year/Month/Day
    ]

    for date_format in date_formats:
        try:
            return datetime.datetime.strptime(date_value, date_format)
        except ValueError:
            continue

    # If we can't parse the date with any format, print it for debugging
    print(f"WARNING: Could not parse date: '{date_value}' (type: {type(date_value)})")
    return None

# 1. Data Loading with error handling
print("Loading data...")

try:
    # Yahoo Finance Bitcoin data
    yahoo_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Yahoo_Finance_Bitcoin.csv")
    print(f"Yahoo data loaded: {len(yahoo_df)} rows")
    print(f"Yahoo columns: {yahoo_df.columns.tolist()}")
    print(f"Yahoo date sample: {yahoo_df['Date'].iloc[0]} (type: {type(yahoo_df['Date'].iloc[0])})")

    # Convert date column with graceful error handling for NaN values
    yahoo_df['Date'] = yahoo_df['Date'].apply(parse_date)
    print(f"Yahoo dates after parsing: {yahoo_df['Date'].isna().sum()} NaN values")

    # Drop rows with null dates
    initial_rows = len(yahoo_df)
    yahoo_df = yahoo_df.dropna(subset=['Date'])
    print(f"Yahoo data after dropping NaN dates: {len(yahoo_df)} rows (dropped {initial_rows - len(yahoo_df)} rows)")

    yahoo_df = yahoo_df.sort_values('Date')

    # Google Trends Bitcoin data
    google_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Google_Trends_bitcoin.csv")
    print(f"Google data loaded: {len(google_df)} rows")
    print(f"Google columns: {google_df.columns.tolist()}")

    # Check if the date column is named 'date' or 'Date'
    date_col = 'date' if 'date' in google_df.columns else 'Date'
    print(f"Google date column found: {date_col}")
    print(f"Google date sample: {google_df[date_col].iloc[0]} (type: {type(google_df[date_col].iloc[0])})")

    # Convert date column with graceful error handling for NaN values
    google_df[date_col] = google_df[date_col].apply(parse_date)
    print(f"Google dates after parsing: {google_df[date_col].isna().sum()} NaN values")

    # Drop rows with null dates
    initial_rows = len(google_df)
    google_df = google_df.dropna(subset=[date_col])
    print(f"Google data after dropping NaN dates: {len(google_df)} rows (dropped {initial_rows - len(google_df)} rows)")

    google_df = google_df.sort_values(date_col)
    google_df = google_df.rename(columns={date_col: 'Date'}) # Ensure consistent column name

    # Twitter sentiment data
    sentiment_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/combined_bitcoin_tweets_with_Sentiment_all_months.csv")
    print(f"Sentiment data loaded: {len(sentiment_df)} rows")
    print(f"Sentiment columns: {sentiment_df.columns.tolist()}")

    # Check if the date column is named 'timestamp_ms' or something else
    if 'timestamp_ms' in sentiment_df.columns:
        date_col = 'timestamp_ms'
    elif 'date' in sentiment_df.columns:
        date_col = 'date'
    elif 'Date' in sentiment_df.columns:
        date_col = 'Date'
    else:
        # Try to find a column that might contain date data
        for col in sentiment_df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        else:
            raise ValueError("Cannot find date column in sentiment data")

    print(f"Sentiment date column found: {date_col}")
    print(f"Sentiment date sample: {sentiment_df[date_col].iloc[0]} (type: {type(sentiment_df[date_col].iloc[0])})")

    # Convert date column with graceful error handling for NaN values
    sentiment_df[date_col] = sentiment_df[date_col].apply(parse_date)
    print(f"Sentiment dates after parsing: {sentiment_df[date_col].isna().sum()} NaN values")

    # Drop rows with null dates
    initial_rows = len(sentiment_df)
    sentiment_df = sentiment_df.dropna(subset=[date_col])
    print(f"Sentiment data after dropping NaN dates: {len(sentiment_df)} rows (dropped {initial_rows - len(sentiment_df)} rows)")

    sentiment_df = sentiment_df.sort_values(date_col)
    sentiment_df = sentiment_df.rename(columns={date_col: 'Date'}) # Ensure consistent column name

    # If all date parsing fails, create a fallback date column
    if len(yahoo_df) == 0 or len(google_df) == 0 or len(sentiment_df) == 0:
        print("\nUsing fallback approach: Creating artificial dates based on row order")

        # Reload original data
        yahoo_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Yahoo_Finance_Bitcoin.csv")
        google_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Google_Trends_bitcoin.csv")
        sentiment_df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/combined_bitcoin_tweets_with_Sentiment_all_months.csv")

        # Create a sequence of dates
        start_date = datetime.datetime(2020, 9, 1)  # Assuming data starts around September 2020
        dates = [start_date + datetime.timedelta(days=i) for i in range(max(len(yahoo_df), len(google_df), len(sentiment_df)))]

        # Add artificial dates to each dataframe
        if len(yahoo_df) > 0:
            yahoo_df['Date'] = dates[:len(yahoo_df)]

        if len(google_df) > 0:
            date_col = 'date' if 'date' in google_df.columns else 'Date'
            google_df[date_col] = dates[:len(google_df)]
            google_df = google_df.rename(columns={date_col: 'Date'})

        if len(sentiment_df) > 0:
            date_col = 'timestamp_ms' if 'timestamp_ms' in sentiment_df.columns else ('date' if 'date' in sentiment_df.columns else 'Date')
            sentiment_df[date_col] = dates[:len(sentiment_df)]
            sentiment_df = sentiment_df.rename(columns={date_col: 'Date'})

        print(f"After fallback approach - Yahoo: {len(yahoo_df)} rows, Google: {len(google_df)} rows, Sentiment: {len(sentiment_df)} rows")

    # If we have valid dates in any dataset, try to standardize the format before merging
    if len(yahoo_df) == 0 and len(google_df) > 0:
        print("\nAttempting to recover Yahoo data using dates from Google...")
        # Check if original Yahoo dates have any patterns we can use
        yahoo_df = pd.read_csv('Yahoo_Finance_Bitcoin.csv')

        # Try to extract dates from existing information
        # This is a guess - the relationship between IDs and dates
        if len(google_df) == len(yahoo_df):
            print("Same number of rows in Yahoo and Google data, attempting to use Google dates...")
            # Create a mapping from index to date
            date_mapping = dict(zip(range(len(google_df)), google_df['Date'].tolist()))

            # Apply dates to Yahoo data
            yahoo_df['Date'] = yahoo_df.index.map(lambda i: date_mapping.get(i))
            print(f"Yahoo data after date recovery: {len(yahoo_df)} rows")

    # Same for sentiment data if needed
    if len(sentiment_df) == 0 and len(google_df) > 0:
        print("\nAttempting to recover Sentiment data using dates from Google...")
        sentiment_df = pd.read_csv('combined_bitcoin_tweets_with_Sentiment_all_months.csv')

        if len(google_df) == len(sentiment_df):
            print("Same number of rows in Sentiment and Google data, attempting to use Google dates...")
            date_mapping = dict(zip(range(len(google_df)), google_df['Date'].tolist()))
            sentiment_df['Date'] = sentiment_df.index.map(lambda i: date_mapping.get(i))
            print(f"Sentiment data after date recovery: {len(sentiment_df)} rows")

    # 3. Data Preparation
    print("Preparing data...")
    # First merging Yahoo and Google
    print(f"Before merge - Yahoo: {len(yahoo_df)} rows, Google: {len(google_df)} rows")
    merged_df = pd.merge(yahoo_df, google_df, on='Date', how='inner')
    print(f"After first merge: {len(merged_df)} rows")

    # Then merging with Twitter (includes sentiment)
    print(f"Before second merge - Merged: {len(merged_df)} rows, Sentiment: {len(sentiment_df)} rows")
    final_df = pd.merge(merged_df, sentiment_df, on='Date', how='inner')
    print(f"After second merge: {len(final_df)} rows")

    # 4. Feature Engineering
    print("Engineering features...")

    # Convert string columns to numeric if needed
    # First check which columns exist in the merged dataframe
    numeric_cols = []
    for col in ['Open', 'Close', 'Volatility (%)', 'Daily Change (indicator)', 'Buzz',
                'positive', 'negative', 'neutral', 'compound']:
        if col in final_df.columns:
            numeric_cols.append(col)
        else:
            print(f"Warning: Column '{col}' not found in merged data")

    print(f"Converting these columns to numeric: {numeric_cols}")
    for col in numeric_cols:
        if final_df[col].dtype == 'object':
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    # Create additional features if all required columns exist
    if 'Close' in final_df.columns:
        # 1. Price momentum (3-day moving average)
        final_df['price_ma3'] = final_df['Close'].rolling(window=3).mean()

        # 2. Price momentum (7-day moving average)
        final_df['price_ma7'] = final_df['Close'].rolling(window=7).mean()

    if 'Buzz' in final_df.columns:
        # 3. Google Trends momentum
        final_df['google_trends_ma3'] = final_df['Buzz'].rolling(window=3).mean()

    if 'compound' in final_df.columns:
        # 4. Sentiment momentum
        final_df['sentiment_ma3'] = final_df['compound'].rolling(window=3).mean()

    # Drop rows with NaN values due to rolling calculations
    final_df_complete = final_df.dropna()
    print(f"Final dataframe after dropping NaN: {len(final_df_complete)} rows")

    # Prepare features and target if the required columns exist
    feature_cols = []
    for col in ['Open', 'Close', 'Volatility (%)', 'Buzz', 'positive', 'negative',
                'neutral', 'compound', 'price_ma3', 'price_ma7', 'google_trends_ma3', 'sentiment_ma3']:
        if col in final_df_complete.columns:
            feature_cols.append(col)

    print(f"Using these features: {feature_cols}")
    X = final_df_complete[feature_cols]

    # Target variable is the "Daily Change(indicator)" column if it exists
    if 'Daily Change (indicator)' in final_df_complete.columns:
        final_df_complete['target'] = final_df_complete['Daily Change (indicator)'].shift(-1)
        final_df_complete = final_df_complete.dropna(subset=['target'])
        y = final_df_complete['target']
        print(f"Target variable prepared: {len(y)} values")
    else:
        print("Warning: 'Daily Change (indicator)' column not found, cannot prepare target variable")

    # 5. Save merged data to a new CSV file
    final_df_complete.to_csv("/content/drive/My Drive/Colab Notebooks/merged_bitcoin_data.csv", index=False)
    print("Merged data saved to 'merged_bitcoin_data.csv'")

    # 6. Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total rows in merged dataset: {len(final_df_complete)}")
    print(f"Date range: {final_df_complete['Date'].min()} to {final_df_complete['Date'].max()}")
    print(f"Total features: {len(feature_cols)}")

except Exception as e:
    import traceback
    print(f"Error: {str(e)}")
    print(traceback.format_exc())
