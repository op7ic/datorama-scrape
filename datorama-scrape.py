import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime
import os
import pandas as pd
import re
import time

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
                # Also try to extract manager name from the link text
                manager_info = {
                    'url': link['href'],
                    'name': link.text.strip() if link.text else ""
                }
                manager_links.append(manager_info)
        
        print(f"Found {len(manager_links)} managers to scrape")
        
        # Scrape each manager's holdings
        for idx, manager_info in enumerate(manager_links, 1):
            try:
                print(f"Scraping manager {idx}/{len(manager_links)}: {manager_info['url']}")
                holdings = self.scrape_manager(manager_info['url'], manager_info.get('name', ''))
                all_holdings.extend(holdings)
                print(f"  Found {len(holdings)} holdings")
                
                # Add a small delay to be polite to the server
                time.sleep(0.5)
            except Exception as e:
                print(f"Error scraping {manager_info['url']}: {e}")
        
        print(f"Total holdings scraped: {len(all_holdings)}")
        
        # Save holdings
        self.save_holdings(all_holdings)
        
        return all_holdings
    
    def scrape_manager(self, manager_link, hint_name=""):
        """Scrape holdings for a specific manager"""
        # Handle both relative and absolute URLs
        if manager_link.startswith('/'):
            manager_url = "https://www.dataroma.com" + manager_link
        elif not manager_link.startswith('http'):
            manager_url = self.base_url + manager_link.lstrip('/')
        else:
            manager_url = manager_link
        
        try:
            response = requests.get(manager_url, timeout=30, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {manager_url}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holdings = []
        
        # Extract manager name - try multiple methods
        manager_name = hint_name
        
        # Method 1: Look for h1 tag
        h1_elem = soup.find('h1')
        if h1_elem and h1_elem.text.strip():
            manager_name = h1_elem.text.strip()
        
        # Method 2: Look for bold text or specific patterns
        if not manager_name or manager_name == "Unknown":
            # Look for text pattern like "Warren Buffett - Berkshire Hathaway"
            for text in soup.stripped_strings:
                if ' - ' in text and 'Portfolio' not in text and 'Period:' not in text:
                    potential_name = text.strip()
                    if len(potential_name) < 100:  # Reasonable length for a name
                        manager_name = potential_name
                        break
        
        print(f"  Manager: {manager_name}")
        
        # Extract portfolio date
        portfolio_date = datetime.now().strftime('%Y-%m-%d')
        date_pattern = re.compile(r'Portfolio date:\s*(\d{1,2}\s+\w+\s+\d{4})')
        for text in soup.stripped_strings:
            match = date_pattern.search(text)
            if match:
                try:
                    date_str = match.group(1)
                    portfolio_date = datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
                    print(f"  Portfolio date: {portfolio_date}")
                except:
                    pass
                break
        
        # Find holdings - look for links to stock.php?sym=
        stock_links = soup.find_all('a', href=re.compile(r'/m/stock\.php\?sym='))
        
        if stock_links:
            print(f"  Found {len(stock_links)} stock links")
            
            # The holdings are typically in a table structure
            # Find the parent table of these links
            for link in stock_links:
                try:
                    # Get the ticker from the URL
                    ticker_match = re.search(r'sym=([A-Z0-9\.]+)', link['href'])
                    if not ticker_match:
                        continue
                    
                    ticker = ticker_match.group(1)
                    
                    # Get the stock name from the link text
                    link_text = link.text.strip()
                    # Parse format like "AAPL - Apple Inc."
                    if ' - ' in link_text:
                        parts = link_text.split(' - ', 1)
                        stock_name = parts[1] if len(parts) > 1 else link_text
                    else:
                        stock_name = link_text
                    
                    # Find the parent row
                    parent_row = link.find_parent('tr')
                    if parent_row:
                        # Extract data from the row
                        cells = parent_row.find_all('td')
                        
                        holding = {
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'portfolio_date': portfolio_date,
                            'manager': manager_name,
                            'stock': stock_name,
                            'ticker': ticker,
                            'portfolio_percent': "0",
                            'shares': "0",
                            'recent_activity': "",
                            'reported_price': None,
                            'current_price': None,
                            'value': None
                        }
                        
                        # Try to extract percentage, shares, and other data
                        # The exact positions might vary, so we'll look for patterns
                        for i, cell in enumerate(cells):
                            cell_text = cell.text.strip()
                            
                            # Portfolio percentage (look for %)
                            if '%' in cell_text and 'portfolio_percent' not in holding:
                                holding['portfolio_percent'] = cell_text.replace('%', '').strip()
                            
                            # Shares (look for large numbers with commas)
                            elif re.match(r'^[\d,]+$', cell_text.replace(',', '')) and int(cell_text.replace(',', '')) > 1000:
                                holding['shares'] = cell_text.replace(',', '')
                            
                            # Recent activity
                            elif cell_text in ['Buy', 'Sell', 'Add', 'Reduce', 'New', 'Sold Out']:
                                holding['recent_activity'] = cell_text
                            
                            # Prices (look for $ or decimal numbers)
                            elif '$' in cell_text or re.match(r'^\d+\.\d{2}$', cell_text):
                                price = self.clean_price(cell_text)
                                if price and not holding['reported_price']:
                                    holding['reported_price'] = price
                                elif price and not holding['current_price']:
                                    holding['current_price'] = price
                        
                        holdings.append(holding)
                        
                except Exception as e:
                    print(f"    Error parsing holding: {e}")
                    continue
        
        else:
            # Alternative method: Look for table with stock data
            # Find all tables and look for the one with holdings
            tables = soup.find_all('table')
            for table in tables:
                # Check if this table contains stock data
                if table.find('a', href=re.compile(r'stock\.php\?sym=')):
                    rows = table.find_all('tr')
                    print(f"  Found table with {len(rows)} rows")
                    
                    for row in rows:
                        # Skip header rows
                        if row.find('th'):
                            continue
                        
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            # Look for stock link in the row
                            stock_link = row.find('a', href=re.compile(r'stock\.php\?sym='))
                            if stock_link:
                                ticker_match = re.search(r'sym=([A-Z0-9\.]+)', stock_link['href'])
                                if ticker_match:
                                    ticker = ticker_match.group(1)
                                    stock_name = stock_link.text.strip()
                                    
                                    if ' - ' in stock_name:
                                        parts = stock_name.split(' - ', 1)
                                        stock_name = parts[1] if len(parts) > 1 else stock_name
                                    
                                    holding = {
                                        'date': datetime.now().strftime('%Y-%m-%d'),
                                        'portfolio_date': portfolio_date,
                                        'manager': manager_name,
                                        'stock': stock_name,
                                        'ticker': ticker,
                                        'portfolio_percent': "0",
                                        'shares': "0",
                                        'recent_activity': "",
                                        'reported_price': None,
                                        'current_price': None
                                    }
                                    
                                    holdings.append(holding)
                    break
        
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
        print(f"Managers tracked: {metadata['managers_tracked']}")
        print(f"Unique stocks: {metadata['unique_stocks']}")

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