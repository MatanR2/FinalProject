!pip install pytrends

from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import pandas as pd
import time

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)
keywords = ["Bitcoin"]

max_retries = 5
for i in range(max_retries):
    try:
        # Build the payload
        pytrends.build_payload(
            kw_list=keywords,
            timeframe='2020-09-29 2021-03-24',
            geo='',
            gprop=''
        )

        # Get the data
        data = pytrends.interest_over_time()

        # If no exception occurred, break the loop
        break
    except TooManyRequestsError:
        print(f"Rate limited. Retry {i+1}/{max_retries} after 10s...")
        time.sleep(10)
else:
    # If the loop completes without a break, raise an error
    raise Exception("Too many requests. All retries failed.")

# Renaming the column for future use
data.rename(columns={'Bitcoin': 'Buzz'}, inplace=True)
print(data.head())
data.to_csv('Google_Trends_bitcoin.csv')
