!pip install pandas vaderSentiment

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Step 1: Load the CSV file
file_path = "/content/drive/My Drive/Colab Notebooks/combined_all_months_data.csv"
df = pd.read_csv(file_path)

# Step 2: Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Step 3: Define a function to calculate sentiment scores
# Returns a dictionary containing sentiment scores:
def get_sentiment_scores(text):
    scores = analyzer.polarity_scores(text)
    return scores

# Step 4: Apply the function to each row in the "text" column
# Stores the results in a new column called "sentiment_scores"
# Each value in this column is a dictionary {pos: x, neg: y, neu: z, compound: w}
df['sentiment_scores'] = df['text'].apply(get_sentiment_scores)

# Step 5: Extract specific sentiment scores into separate columns
df['positive'] = df['sentiment_scores'].apply(lambda x: x.get('pos', None))
df['negative'] = df['sentiment_scores'].apply(lambda x: x.get('neg', None))
df['neutral'] = df['sentiment_scores'].apply(lambda x: x.get('neu', None))
df['compound'] = df['sentiment_scores'].apply(lambda x: x.get('compound', None))

# Step 6: divide into two df (Bitcoin and Ethereum)
df_bitcoin = df[df['text'].str.contains(r'bitcoin|\$btc', case=False, na=False)]
df_ethereum = df[df['text'].str.contains(r'ethereum|\$eth', case=False, na=False)]


# Step 7: Create new CSV for each currency and combine days
df_combined_bitcoin = df_bitcoin.groupby('timestamp_ms').agg({
    'positive': 'mean',
    'negative': 'mean',
    'neutral': 'mean',
    'compound': 'mean'
}).reset_index()
df_combined_ethereum = df_ethereum.groupby('timestamp_ms').agg({
    'positive': 'mean',
    'negative': 'mean',
    'neutral': 'mean',
    'compound': 'mean'
}).reset_index()
df_combined_bitcoin['timestamp_ms'] = pd.to_datetime(df_combined_bitcoin['timestamp_ms'], format='%d/%m/%Y')
df_combined_bitcoin = df_combined_bitcoin.sort_values(by='timestamp_ms')
df_combined_ethereum['timestamp_ms'] = pd.to_datetime(df_combined_ethereum['timestamp_ms'], format='%d/%m/%Y')
df_combined_ethereum = df_combined_ethereum.sort_values(by='timestamp_ms')


# Step 8: Add a "sentiment" column
def classify_sentiment(compound):
    if compound >= 0.1135:
        return 1  # Positive or Neutral
    else:
        return -1  # Negative

df_combined_bitcoin['sentiment'] = df_combined_bitcoin['compound'].apply(classify_sentiment)
df_combined_ethereum['sentiment'] = df_combined_ethereum['compound'].apply(classify_sentiment)


# Step 9: Save the updated Dataframes to a new CSV files
output_path = "/content/drive/My Drive/Colab Notebooks/combined_bitcoin_tweets_with_Sentiment_all_months.csv"
df_combined_bitcoin.to_csv(output_path, index=False)
output_path = "/content/drive/My Drive/Colab Notebooks/combined_ethereum_tweets_with_Sentiment_all_months.csv"
df_combined_ethereum.to_csv(output_path, index=False)

'''
# Sort the bitcoin dataframe by compound score (ascending order)
df_bitcoin_sorted = df_bitcoin.sort_values(by='compound', ascending=True)
df_ethereum_sorted = df_ethereum.sort_values(by='compound', ascending=True)
bitcoin_sorted_path = "/content/drive/My Drive/Colab Notebooks/bitcoin_tweets_sorted_by_compound.csv"
df_bitcoin_sorted.to_csv(bitcoin_sorted_path, index=False)
ethereum_sorted_path = "/content/drive/My Drive/Colab Notebooks/ethereum_tweets_sorted_by_compound.csv"
df_ethereum_sorted.to_csv(ethereum_sorted_path, index=False)

output_path = "/content/drive/My Drive/Colab Notebooks/bitcoin_tweets_with_Sentiment_all_months.csv"
df_bitcoin.to_csv(output_path, index=False)
output_path = "/content/drive/My Drive/Colab Notebooks/ethereum_tweets_with_Sentiment_all_months.csv"
df_ethereum.to_csv(output_path, index=False)
'''
