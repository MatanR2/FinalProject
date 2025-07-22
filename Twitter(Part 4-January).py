json_file4 = "/content/drive/My Drive/Colab Notebooks/january.json"
pip install ijson
import json #Standard library for working with JSON data
import csv #Standard library for reading and writing CSV files.
import ijson #A library for parsing large JSON files in an iterative (streaming) manner, instead of loading everything into memory.
import os #Provides functions for interacting with the operating system, like creating directories and working with file paths.

def process_json_to_csv_chunks(input_file_path, output_dir, chunk_size=100000, prefix="january_"):
    """
    Process a large JSON array file, convert each object to CSV format, and save in chunks.
    The function processes a large JSON file by converting its objects into CSV format in chunks, reducing memory usage.

    Arguments:
        input_file_path (str): Path to the input JSON file.
        output_dir (str): Directory where CSV chunk files will be saved.
        chunk_size (int): Number of rows per CSV chunk file.
        prefix (str): Prefix for naming the output CSV files

    Returns:
        int: Number of chunks created
    """
    # Create output directory if it doesn't exist
    # exist_ok=True ensures no error occurs if the directory already exists.
    os.makedirs(output_dir, exist_ok=True)

    chunk_count = 0 #Tracks the number of chunk files written.
    object_count = 0 #Tracks the number of JSON objects processed.
    current_chunk_rows = [] #Stores rows temporarily before writing them to a CSV file.

    # Column headers (based on the example data structure)
    columns = ['id', 'text', 'keywords', 'timestamp_ms', 'followers_count',
               'urls', 'is_retweet', 'retweeted_id']

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:#Opens the JSON file for reading ('r' mode) with UTF-8 encoding.
            # Create an iterator over JSON objects in the array
            objects = ijson.items(f, 'item') #ijson reads only one object at a time, making it memory-efficient for large files.

            for obj in objects: #Iterates over each JSON object in the file one at a time.
                # Create a dictionary for the current row
                row = {}

                # Process each field from the JSON object
                for column in columns:
                    if column in obj:
                        # Handle array fields (keywords, urls)
                        if column in ['keywords', 'urls'] and isinstance(obj[column], list):
                            # Join array elements with a separator
                            row[column] = '|'.join(str(item) for item in obj[column])
                        else:
                            row[column] = obj[column]
                    else:
                        row[column] = ''  # Default value for missing fields

                current_chunk_rows.append(row) # Adds the processed row (dictionary) to current_chunk_rows.
                object_count += 1 # Increments object_count to keep track of the total number of processed JSON objects.

                # When we reach chunk_size, write to CSV file and start a new chunk
                if len(current_chunk_rows) >= chunk_size:
                    output_file = os.path.join(output_dir, f"{prefix}{chunk_count}.csv")

                    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(current_chunk_rows)

                    print(f"Wrote chunk {chunk_count} with {len(current_chunk_rows)} rows to {output_file}")

                    # Reset for next chunk
                    current_chunk_rows = []
                    chunk_count += 1

                # Progress reporting
                if object_count % 10000 == 0:
                    print(f"Processed {object_count} objects...")

        # Write any remaining rows to the final chunk
        if current_chunk_rows:
            output_file = os.path.join(output_dir, f"{prefix}{chunk_count}.csv")

            with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(current_chunk_rows)

            print(f"Wrote final chunk {chunk_count} with {len(current_chunk_rows)} rows to {output_file}")
            chunk_count += 1
    # Error handling
    except FileNotFoundError: # Catches errors if the input file does not exist.
        print(f"Error: File {input_file_path} not found")
    except json.JSONDecodeError as e:   # Catches errors if the JSON file is malformed.
        print(f"Error decoding JSON: {e}")
    except Exception as e:  # Catches any other unexpected errors.
        print(f"Unexpected error: {e}")

    return chunk_count

    input_file_path = json_file4
    output_dir = "/content/drive/My Drive/Colab Notebooks/csv_chunks_JANUARY"  # Directory to store CSV chunk files

    print(f"Processing {input_file_path} in chunks of 100,000 rows...")
    chunks_created = process_json_to_csv_chunks(input_file_path, output_dir)

    print(f"Processing complete. Created {chunks_created} CSV files in {output_dir}")

!pip install langdetect
import csv
import os
import datetime
import glob
from time import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

def is_english_text(text):
    """
    Check if the given text is in English.

    Arguments:
        text (str): Text to check

    Returns:
        bool: True if text is detected as English, False otherwise
    """
    if not text or len(text.strip()) < 3:
        return False

    try:
        # Create a cleaned version for language detection without modifying original
        clean_text = text

        # Remove @mentions and URLs which can interfere with detection
        words = clean_text.split()
        filtered_words = [word for word in words if not word.startswith('@') and not word.startswith('http')]
        clean_text = ' '.join(filtered_words)

        # If text starts with RT, remove just the RT part for language detection
        if clean_text.startswith('RT '):
            clean_text = clean_text[3:].strip()

        if len(clean_text.strip()) < 3:
            return False

        detected_lang = detect(clean_text)
        return detected_lang == 'en'
    except LangDetectException:
        # If detection fails, assume it's not English
        return False

def first_pass_filter(input_file, temp_file):
    """
    First pass: Apply fast filters (retweets and crypto keywords).

    Arguments:
        input_file (str): Path to the input CSV file
        temp_file (str): Path to save the temporarily filtered CSV file

    Returns:
        tuple: (rows_read, rows_written, filter_stats)
    """
    # Crypto keywords to filter by (case insensitive)
    crypto_keywords = ['btc', '$btc', 'bitcoin', 'eth', '$eth', 'ethereum']

    rows_read = 0
    rows_written = 0

    filter_stats = {
        'retweets_filtered': 0,
        'no_crypto_keywords': 0
    }

    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        all_fieldnames = reader.fieldnames

        # Make sure we have all required columns
        required_fields = ['text', 'is_retweet', 'keywords', 'timestamp_ms']
        if not all(field in all_fieldnames for field in required_fields):
            raise ValueError(f"Missing required columns in {input_file}")

        # Prepare temporary output file
        with open(temp_file, 'w', newline='', encoding='utf-8') as temp_outfile:
            writer = csv.DictWriter(temp_outfile, fieldnames=all_fieldnames)
            writer.writeheader()

            for row in reader:
                rows_read += 1

                # Filter 1: Skip if text starts with "RT" or is_retweet is TRUE
                if row['text'].startswith('RT') or row['is_retweet'].lower() == 'true':
                    filter_stats['retweets_filtered'] += 1
                    continue

                # Filter 2: Skip if keywords field doesn't contain any of the crypto terms
                if 'keywords' in row and row['keywords']:
                    # Convert keywords to lowercase for case-insensitive comparison
                    keywords_lower = row['keywords'].lower()
                    # Check if any of the crypto keywords is present
                    if not any(keyword in keywords_lower for keyword in crypto_keywords):
                        filter_stats['no_crypto_keywords'] += 1
                        continue
                else:
                    # No keywords in this row, skip it
                    filter_stats['no_crypto_keywords'] += 1
                    continue

                # Write the row that passed the fast filters
                writer.writerow(row)
                rows_written += 1

    return rows_read, rows_written, filter_stats

def second_pass_filter(temp_file, output_file):
    """
    Second pass: Apply expensive language detection and final processing.

    Arguments:
        temp_file (str): Path to the temporarily filtered CSV file
        output_file (str): Path to save the final processed CSV file

    Returns:
        tuple: (rows_read, rows_written, filter_stats)
    """
    # Columns to remove after all processing
    columns_to_remove = ['id', 'followers_count', 'urls', 'is_retweet', 'retweeted_id']

    rows_read = 0
    rows_written = 0

    filter_stats = {
        'non_english_filtered': 0,
        'timestamp_conversion_errors': 0
    }

    with open(temp_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        all_fieldnames = reader.fieldnames

        # Create a list for filtered rows
        filtered_rows = []

        for row in reader:
            rows_read += 1

            # Filter: Skip if text is not in English
            if not is_english_text(row['text']):
                filter_stats['non_english_filtered'] += 1
                continue

            # Convert timestamp_ms to readable date
            if 'timestamp_ms' in row and row['timestamp_ms']:
                try:
                    # Convert milliseconds since epoch to seconds and create a datetime object
                    timestamp = int(row['timestamp_ms']) / 1000
                    date_obj = datetime.datetime.fromtimestamp(timestamp)
                    # Format as dd/mm/yyyy
                    row['timestamp_ms'] = date_obj.strftime('%d/%m/%Y')
                except (ValueError, TypeError):
                    filter_stats['timestamp_conversion_errors'] += 1
                    # Keep original if conversion fails
                    pass

            # Add the filtered and processed row
            filtered_rows.append(row)
            rows_written += 1

    # Create new fieldnames list excluding columns to remove
    final_fieldnames = [field for field in all_fieldnames if field not in columns_to_remove]

    # Write the filtered data to the output file, excluding removed columns
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=final_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(filtered_rows)

    return rows_read, rows_written, filter_stats

def process_csv_file_two_pass(input_file, output_file):
    """
    Process a CSV file using two-pass filtering for better performance:
    Pass 1: Fast filters (retweets, crypto keywords)
    Pass 2: Expensive filters (language detection) + final processing

    Arguments:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the processed CSV file

    Returns:
        tuple: (total_rows_read, rows_written, combined_filter_stats)
    """
    # Create temporary file name
    temp_file = output_file + '.temp'

    try:
        print(f"    Pass 1: Applying fast filters...")
        start_pass1 = time()
        rows_read, pass1_written, pass1_stats = first_pass_filter(input_file, temp_file)
        end_pass1 = time()

        print(f"    Pass 1 complete: {rows_read} -> {pass1_written} rows ({(end_pass1 - start_pass1):.2f}s)")
        print(f"      - Retweets filtered: {pass1_stats['retweets_filtered']}")
        print(f"      - No crypto keywords: {pass1_stats['no_crypto_keywords']}")

        print(f"    Pass 2: Applying language detection...")
        start_pass2 = time()
        pass2_read, rows_written, pass2_stats = second_pass_filter(temp_file, output_file)
        end_pass2 = time()

        print(f"    Pass 2 complete: {pass2_read} -> {rows_written} rows ({(end_pass2 - start_pass2):.2f}s)")
        print(f"      - Non-English filtered: {pass2_stats['non_english_filtered']}")

        # Combine filter statistics
        combined_stats = {
            'retweets_filtered': pass1_stats['retweets_filtered'],
            'no_crypto_keywords': pass1_stats['no_crypto_keywords'],
            'non_english_filtered': pass2_stats['non_english_filtered'],
            'timestamp_conversion_errors': pass2_stats['timestamp_conversion_errors']
        }

        return rows_read, rows_written, combined_stats

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def process_all_csv_files(input_dir, output_dir, file_prefix="january_"):
    """
    Process all CSV files in the input directory with the given prefix using two-pass filtering.

    Arguments:
        input_dir (str): Directory containing CSV files
        output_dir (str): Directory to save processed CSV files
        file_prefix (str): Prefix of files to process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all matching CSV files
    pattern = os.path.join(input_dir, f"{file_prefix}*.csv")
    csv_files = glob.glob(pattern)

    total_files = len(csv_files)
    total_rows_read = 0
    total_rows_written = 0

    # Aggregate filter statistics
    total_filter_stats = {
        'retweets_filtered': 0,
        'non_english_filtered': 0,
        'no_crypto_keywords': 0,
        'timestamp_conversion_errors': 0
    }

    print(f"Found {total_files} CSV files to process")

    start_time = time()

    for i, input_file in enumerate(csv_files, 1):
        # Create output filename
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"filtered_{filename}")

        print(f"Processing file {i}/{total_files}: {filename}")
        try:
            rows_read, rows_written, filter_stats = process_csv_file_two_pass(input_file, output_file)
            total_rows_read += rows_read
            total_rows_written += rows_written

            # Aggregate filter statistics
            for key in total_filter_stats:
                total_filter_stats[key] += filter_stats[key]

            print(f"  - Final result: {rows_read} -> {rows_written} rows")
            print(f"  - Overall filter rate: {(rows_read - rows_written) / rows_read * 100:.2f}%")
        except Exception as e:
            print(f"  - Error processing {filename}: {e}")

    end_time = time()
    elapsed_time = end_time - start_time

    print("\nProcessing complete!")
    print(f"Total files processed: {total_files}")
    print(f"Total rows read: {total_rows_read}")
    print(f"Total rows written: {total_rows_written}")
    print(f"\nFilter breakdown:")
    print(f"  - Retweets filtered: {total_filter_stats['retweets_filtered']}")
    print(f"  - Non-English filtered: {total_filter_stats['non_english_filtered']}")
    print(f"  - No crypto keywords: {total_filter_stats['no_crypto_keywords']}")
    print(f"  - Timestamp errors: {total_filter_stats['timestamp_conversion_errors']}")
    print(f"Overall filter rate: {(total_rows_read - total_rows_written) / total_rows_read * 100:.2f}%")
    print(f"Total processing time: {elapsed_time:.2f} seconds")

input_dir = "/content/drive/My Drive/Colab Notebooks/csv_chunks_JANUARY"  # Directory containing the CSV files
output_dir = "/content/drive/My Drive/Colab Notebooks/csv_chunks_JANUARY_filtered"  # Directory to save filtered CSV files

print("Starting CSV filtering and processing...")
process_all_csv_files(input_dir, output_dir)

import csv
import os
import glob
from time import time
import datetime

def combine_and_sort_csv_files(input_dir, output_file, date_column="timestamp_ms", file_pattern="filtered_january_*.csv"):
    """
    Combines multiple CSV files into a single CSV file and sorts by date.

    Arguments:
        input_dir (str): Directory containing the CSV files to combine
        output_file (str): Path to the output combined CSV file
        date_column (str): Name of the column containing the date to sort by
        file_pattern (str): Glob pattern to match files to combine

    Returns:
        tuple: (files_processed, total_rows) statistics
    """
    start_time = time()
    pattern = os.path.join(input_dir, file_pattern)
    csv_files = sorted(glob.glob(pattern))

    total_files = len(csv_files)
    if total_files == 0:
        print(f"No files found matching pattern '{file_pattern}' in '{input_dir}'")
        return 0, 0

    print(f"Found {total_files} CSV files to combine")

    # Get headers from the first file and find the index of the date column
    with open(csv_files[0], 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)

        if date_column not in headers:
            raise ValueError(f"Date column '{date_column}' not found in CSV headers")

        date_column_index = headers.index(date_column)

    # Read all rows from all files into memory (with their headers)
    all_rows = []
    total_rows = 0

    for i, csv_file in enumerate(csv_files, 1):
        file_rows = 0
        print(f"Reading file {i}/{total_files}: {os.path.basename(csv_file)}")

        try:
            with open(csv_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip the header row

                for row in reader:
                    all_rows.append(row)
                    file_rows += 1
                    total_rows += 1

            print(f"  - Read {file_rows} rows")

        except Exception as e:
            print(f"  - Error processing {csv_file}: {e}")

    print(f"\nSorting {total_rows} rows by date...")

    # Define a function to convert date string to sortable format
    def get_date_key(row):
        date_str = row[date_column_index]
        try:
            # Parse the date in format dd/mm/yyyy
            return datetime.datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            # If date parsing fails, return a minimum date to put it at the beginning
            return datetime.datetime.min

    # Sort all rows by the date
    all_rows.sort(key=get_date_key)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Write the sorted data to the output file
    print(f"Writing sorted data to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)  # Write the header row
        writer.writerows(all_rows)  # Write all sorted rows

    end_time = time()
    elapsed_time = end_time - start_time

    print("\nCombining and sorting complete!")
    print(f"Total files processed: {total_files}")
    print(f"Total rows combined and sorted: {total_rows}")
    print(f"Output file: {output_file}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")

    return total_files, total_rows

input_dir = "/content/drive/My Drive/Colab Notebooks/csv_chunks_JANUARY_filtered"
output_file = "/content/drive/My Drive/Colab Notebooks/combined_sorted_january_data.csv"

print("Starting CSV file combination and sorting...")
combine_and_sort_csv_files(input_dir, output_file)
