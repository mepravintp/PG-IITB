import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Example: Scrape match list from ESPN Cricinfo World Cup page
# Update URL for each tournament as needed
TOURNAMENT_URLS = [
    # Add URLs for ODI/T20, men/women world cups from 2020 onwards
    'https://www.espncricinfo.com/series/icc-men-s-t20-world-cup-2022-1297720',
    'https://www.espncricinfo.com/series/icc-women-s-t20-world-cup-2020-1219022',
    # Add more as needed
]

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')


def fetch_match_list(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    matches = []
    # Example: Find match links and basic info
    for match in soup.find_all('a', href=True):
        if '/match/' in match['href']:
            matches.append({
                'match_url': 'https://www.espncricinfo.com' + match['href'],
                'match_title': match.text.strip()
            })
    return matches


def main():
    all_matches = []
    for url in TOURNAMENT_URLS:
        print(f'Scraping tournament: {url}')
        matches = fetch_match_list(url)
        for m in matches:
            m['tournament_url'] = url
        all_matches.extend(matches)
    df = pd.DataFrame(all_matches)
    out_path = os.path.join(RAW_DATA_DIR, 'matches_list.csv')
    df.to_csv(out_path, index=False)
    print(f'Saved match list to {out_path}')

if __name__ == '__main__':
    main()
