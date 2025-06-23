# datorama-scrape.py
import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime
import os
import pandas as pd

class DataromaScraper:
    def __init__(self):
        self.base_url = "https://www.dataroma.com/m/"
        self.holdings_dir = "holdings"
        self.ensure_directories()
        
        # Set headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.holdings_dir, exist_ok=True)
        os.makedirs("analysis", exist_ok=True)
    
    def scrape_dataroma(self):
        """Scrape Dataroma for all superinvestor holdings"""
        # Get list of managers
        managers_url = self.base_url + "managers.php"
        print(f"Fetching managers from: {managers_url}")
        
        try:
            response = requests.get(managers_url, timeout=30, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching managers page: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        all_holdings = []
        manager_links = []
        
        # Extract manager links - look for links containing 'holdings.php?m='
        for link in soup.find_all('a', href=True):
            if 'holdings.php?m=' in link['href']:
                manager_links.append(link['href'])
        
        print(f"Found {len(manager_links)} managers to scrape")
        
        # Scrape each manager's holdings
        for idx, manager_link in enumerate(manager_links, 1):
            try:
                print(f"Scraping manager {idx}/{len(manager_links)}: {manager_link}")
                holdings = self.scrape_manager(manager_link)
                all_holdings.extend(holdings)
                print(f"  Found {len(holdings)} holdings")
            except Exception as e:
                print(f"Error scraping {manager_link}: {e}")
        
        print(f"Total holdings scraped: {len(all_holdings)}")
        
        # Save holdings
        self.save_holdings(all_holdings)
        
        return all_holdings
    
    def scrape_manager(self, manager_link):
        """Scrape holdings for a specific manager"""
        manager_url = self.base_url + manager_link
        
        try:
            response = requests.get(manager_url, timeout=30, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {manager_url}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holdings = []
        
        # Extract manager name - try multiple methods
        manager_name = "Unknown"
        h1_elem = soup.find('h1')
        if h1_elem:
            manager_name = h1_elem.text.strip()
        else:
            # Try to find manager name in title or other elements
            title_elem = soup.find('title')
            if title_elem and ' - ' in title_elem.text:
                manager_name = title_elem.text.split(' - ')[0].strip()
        
        print(f"  Manager: {manager_name}")
        
        # Extract portfolio date
        date_elem = soup.find(string=lambda string: string and 'Portfolio date:' in string)
        portfolio_date = datetime.now().strftime('%Y-%m-%d')
        if date_elem:
            try:
                date_str = date_elem.split('Portfolio date:')[1].strip()
                portfolio_date = datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
            except:
                pass
        
        # Find holdings table - try multiple approaches
        table = soup.find('table', {'id': 'grid'})
        if not table:
            # Try finding table by class or other attributes
            table = soup.find('table', class_='portfolio-table') or \
                   soup.find('table', class_='holdings-table') or \
                   soup.find('table')
        
        if table:
            # Find all rows, handling both tbody and direct tr children
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
            else:
                rows = table.find_all('tr')
            
            # Skip header row(s)
            data_rows = []
            for row in rows:
                # Check if it's a data row (has td elements, not th)
                if row.find('td'):
                    data_rows.append(row)
            
            print(f"  Found {len(data_rows)} data rows in table")
            
            for row in data_rows:
                cols = row.find_all('td')
                if len(cols) >= 6:  # Minimum expected columns
                    try:
                        holding = {
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'portfolio_date': portfolio_date,
                            'manager': manager_name,
                            'stock': cols[0].text.strip(),
                            'ticker': cols[1].text.strip() if len(cols) > 1 else "",
                            'portfolio_percent': cols[2].text.strip().replace('%', '') if len(cols) > 2 else "0",
                            'shares': cols[3].text.strip().replace(',', '') if len(cols) > 3 else "0",
                            'recent_activity': cols[4].text.strip() if len(cols) > 4 else "",
                            'reported_price': self.clean_price(cols[5].text.strip()) if len(cols) > 5 else None,
                            'current_price': self.clean_price(cols[6].text.strip()) if len(cols) > 6 else None
                        }
                        
                        # Only add if we have valid ticker and stock name
                        if holding['ticker'] and holding['stock']:
                            holdings.append(holding)
                    except Exception as e:
                        print(f"    Error parsing row: {e}")
        else:
            print(f"  No holdings table found for {manager_name}")
        
        return holdings
    
    def clean_price(self, price_str):
        """Clean price string to float"""
        if not price_str:
            return None
        try:
            # Remove dollar sign, commas, and any whitespace
            cleaned = price_str.replace('$', '').replace(',', '').strip()
            if cleaned and cleaned != '-' and cleaned != 'N/A':
                return float(cleaned)
        except (ValueError, AttributeError):
            pass
        return None
    
    def save_holdings(self, all_holdings):
        """Save holdings with smart file management"""
        today = datetime.now().strftime('%Y%m%d')
        
        # Check if we have any holdings
        if not all_holdings:
            print("No holdings found to save")
            return
        
        # Save daily snapshot in holdings folder
        daily_filename = os.path.join(self.holdings_dir, f'holdings_{today}.csv')
        df = pd.DataFrame(all_holdings)
        df.to_csv(daily_filename, index=False)
        
        # Save as latest in root for easy access
        df.to_csv('latest_holdings.csv', index=False)
        
        # Create a lightweight metadata file
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_holdings': len(all_holdings),
            'managers_tracked': len(df['manager'].unique()) if 'manager' in df.columns else 0,
            'unique_stocks': len(df['ticker'].unique()) if 'ticker' in df.columns else 0,
            'file_location': daily_filename
        }
        
        with open('scrape_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(all_holdings)} holdings to {daily_filename}")

def main():
    scraper = DataromaScraper()
    
    # Add a simple connectivity test
    test_url = "https://www.dataroma.com/m/home.php"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(test_url, timeout=10, headers=headers)
        if response.status_code == 200:
            print(f"Successfully connected to Dataroma (status: {response.status_code})")
        else:
            print(f"Warning: Dataroma returned status {response.status_code}")
    except Exception as e:
        print(f"Error connecting to Dataroma: {e}")
        return
    
    # Run the scraper
    scraper.scrape_dataroma()

if __name__ == "__main__":
    main()