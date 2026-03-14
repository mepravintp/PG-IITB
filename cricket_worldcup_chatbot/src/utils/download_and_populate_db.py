import requests
import pandas as pd
import sqlite3
import os
import zipfile
import glob
import yaml

# Example open source cricket event data (replace with actual source)
EVENT_DATA_URL = 'https://cricsheet.org/downloads/'  # Cricsheet provides ball-by-ball data
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cricket_events.db')


def download_event_data():
    # Download ZIP file from GitHub
    zip_url = 'https://github.com/hmnshudhmn24/international-t20-cricket-dataset-2005-2025/raw/main/t20s.zip'
    zip_path = os.path.join(RAW_DATA_DIR, 't20s.zip')
    print(f'Downloading {zip_url}')
    r = requests.get(zip_url)
    with open(zip_path, 'wb') as f:
        f.write(r.content)
    print(f'Saved to {zip_path}')

    # Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
    print(f'Extracted ZIP to {RAW_DATA_DIR}')

    # Kaggle download requires authentication, so instruct user
    kaggle_csv_path = os.path.join(RAW_DATA_DIR, 'international_cricket_stats_2026.csv')
    print('To download Kaggle dataset, please manually download from:')
    print('https://www.kaggle.com/datasets/atifkhan12/international-cricket-stats-2026-odi-and-t20')
    print(f'and place the CSV file as {kaggle_csv_path}')


def parse_event_data():
    # Try to parse extracted YAML files from GitHub ZIP
    yaml_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.yaml'))
    if yaml_files:
        print(f'Parsing {len(yaml_files)} YAML files from {RAW_DATA_DIR}')
        records = []
        for yfile in yaml_files:
            with open(yfile, 'r') as f:
                data = yaml.safe_load(f)
                # Example: extract deliveries
                if 'innings' in data:
                    for inn in data['innings']:
                        for key in inn:
                            for delivery in inn[key]['deliveries']:
                                for ball, event in delivery.items():
                                    event_record = {'ball': ball}
                                    event_record.update(event)
                                    records.append(event_record)
        df = pd.DataFrame(records)
        return df
    # Try Kaggle CSV
    kaggle_csv_path = os.path.join(RAW_DATA_DIR, 'international_cricket_stats_2026.csv')
    if os.path.exists(kaggle_csv_path):
        print(f'Parsing {kaggle_csv_path}')
        df = pd.read_csv(kaggle_csv_path)
        return df
    else:
        print('No event data found. Please download from Kaggle or GitHub as instructed.')
        return None


def populate_sqlite_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('events', conn, if_exists='replace', index=False)
    conn.close()
    print(f'Populated SQLite DB at {DB_PATH}')


def main():
    download_event_data()
    df = parse_event_data()
    if df is not None:
        populate_sqlite_db(df)
    else:
        print('No event data parsed.')

if __name__ == '__main__':
    main()
