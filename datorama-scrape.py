import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime, timedelta
import os
import pandas as pd
import re
import time
from collections import defaultdict

class DataromaScraper:
    def __init__(self):
        self.base_url = "https://www.dataroma.com/m/"
        self.holdings_dir = "holdings"
        self.cache_dir = "cache"
        self.ensure_directories()
        self.stock_cache = self.load_stock_cache()
        
        # Set headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.holdings_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs("analysis", exist_ok=True)
    
    def load_stock_cache(self):
        """Load cached stock data"""
        cache_file = os.path.join(self.cache_dir, 'stock_data_cache.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_stock_cache(self):
        """Save stock cache to disk"""
        cache_file = os.path.join(self.cache_dir, 'stock_data_cache.json')
        with open(cache_file, 'w') as f:
            json.dump(self.stock_cache, f, indent=2)
    
    def should_refresh_stock(self, ticker):
        """Check if stock data should be refreshed"""
        if ticker not in self.stock_cache:
            return True
        
        # Refresh if data is older than 7 days
        cached_data = self.stock_cache[ticker]
        if 'last_updated' in cached_data:
            last_updated = datetime.fromisoformat(cached_data['last_updated'])
            if datetime.now() - last_updated > timedelta(days=7):
                return True
        
        # Refresh if key data is missing
        required_fields = ['market_cap', 'sector', 'current_price']
        if any(field not in cached_data or not cached_data[field] for field in required_fields):
            return True
        
        return False
    
    def scrape_stock_page(self, ticker):
        """Scrape detailed information from individual stock page"""
        # Check if we need to refresh
        if not self.should_refresh_stock(ticker):
            print(f"    Using cached data for {ticker}")
            return self.stock_cache[ticker]
        
        stock_url = f"{self.base_url}stock.php?sym={ticker}"
        print(f"    Fetching fresh data for {ticker}")
        
        try:
            response = requests.get(stock_url, timeout=10, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            stock_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat()
            }
            
            # Extract company name from title or h1
            h1 = soup.find('h1')
            if h1:
                stock_data['company_name'] = h1.text.strip()
            
            # Look for key-value pairs in the page
            # Dataroma often uses patterns like "Market Cap: $XXX"
            text_content = soup.get_text()
            
            # Market Cap
            market_cap_match = re.search(r'Market Cap[:\s]+\$?([\d,]+\.?\d*)\s*([BMK])?', text_content, re.IGNORECASE)
            if market_cap_match:
                value = float(market_cap_match.group(1).replace(',', ''))
                multiplier = market_cap_match.group(2)
                if multiplier:
                    if multiplier.upper() == 'B':
                        value *= 1_000_000_000
                    elif multiplier.upper() == 'M':
                        value *= 1_000_000
                    elif multiplier.upper() == 'K':
                        value *= 1_000
                stock_data['market_cap'] = value
                stock_data['market_cap_display'] = market_cap_match.group(0)
            
            # Sector
            sector_match = re.search(r'Sector[:\s]+([^\n]+)', text_content, re.IGNORECASE)
            if sector_match:
                stock_data['sector'] = sector_match.group(1).strip()
            
            # Industry
            industry_match = re.search(r'Industry[:\s]+([^\n]+)', text_content, re.IGNORECASE)
            if industry_match:
                stock_data['industry'] = industry_match.group(1).strip()
            
            # Current Price - look for various patterns
            price_patterns = [
                r'Current Price[:\s]+\$?([\d,]+\.?\d*)',
                r'Price[:\s]+\$?([\d,]+\.?\d*)',
                r'\$?([\d,]+\.?\d*)\s+(?:USD|usd)',
            ]
            
            for pattern in price_patterns:
                price_match = re.search(pattern, text_content)
                if price_match:
                    stock_data['current_price'] = float(price_match.group(1).replace(',', ''))
                    break
            
            # 52 Week High/Low
            high_match = re.search(r'52\s*Week\s*High[:\s]+\$?([\d,]+\.?\d*)', text_content, re.IGNORECASE)
            if high_match:
                stock_data['52_week_high'] = float(high_match.group(1).replace(',', ''))
            
            low_match = re.search(r'52\s*Week\s*Low[:\s]+\$?([\d,]+\.?\d*)', text_content, re.IGNORECASE)
            if low_match:
                stock_data['52_week_low'] = float(low_match.group(1).replace(',', ''))
            
            # P/E Ratio
            pe_match = re.search(r'P/E\s*(?:Ratio)?[:\s]+([\d,]+\.?\d*)', text_content, re.IGNORECASE)
            if pe_match:
                stock_data['pe_ratio'] = float(pe_match.group(1).replace(',', ''))
            
            # Dividend Yield
            div_match = re.search(r'Dividend\s*(?:Yield)?[:\s]+([\d,]+\.?\d*)%?', text_content, re.IGNORECASE)
            if div_match:
                stock_data['dividend_yield'] = float(div_match.group(1).replace(',', ''))
            
            # Also check for structured data in tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].text.strip().lower()
                        value = cells[1].text.strip()
                        
                        if 'market cap' in label and 'market_cap' not in stock_data:
                            # Parse market cap
                            cap_value = self.parse_market_cap(value)
                            if cap_value:
                                stock_data['market_cap'] = cap_value
                                stock_data['market_cap_display'] = value
                        
                        elif 'sector' in label and 'sector' not in stock_data:
                            stock_data['sector'] = value
                        
                        elif 'industry' in label and 'industry' not in stock_data:
                            stock_data['industry'] = value
                        
                        elif 'price' in label and 'current_price' not in stock_data:
                            price = self.clean_price(value)
                            if price:
                                stock_data['current_price'] = price
            
            # Update cache
            self.stock_cache[ticker] = stock_data
            
            # Small delay to be polite
            time.sleep(0.3)
            
            return stock_data
            
        except Exception as e:
            print(f"    Error fetching stock page for {ticker}: {e}")
            # Return cached data if available, otherwise empty
            return self.stock_cache.get(ticker, {'ticker': ticker})
    
    def parse_market_cap(self, cap_str):
        """Parse market cap string to number"""
        try:
            # Remove $ and spaces
            cap_str = cap_str.replace('$', '').replace(' ', '').strip()
            
            # Handle different formats
            if 'T' in cap_str or 'trillion' in cap_str.lower():
                return float(re.search(r'([\d.]+)', cap_str).group(1)) * 1_000_000_000_000
            elif 'B' in cap_str or 'billion' in cap_str.lower():
                return float(re.search(r'([\d.]+)', cap_str).group(1)) * 1_000_000_000
            elif 'M' in cap_str or 'million' in cap_str.lower():
                return float(re.search(r'([\d.]+)', cap_str).group(1)) * 1_000_000
            else:
                # Try to parse as regular number
                return float(cap_str.replace(',', ''))
        except:
            return None
    
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
        
        # Extract manager links
        for link in soup.find_all('a', href=True):
            if 'holdings.php?m=' in link['href']:
                manager_info = {
                    'url': link['href'],
                    'name': link.text.strip() if link.text else ""
                }
                manager_links.append(manager_info)
        
        print(f"Found {len(manager_links)} managers to scrape")
        
        # First pass: Get all holdings
        for idx, manager_info in enumerate(manager_links, 1):
            try:
                print(f"\nScraping manager {idx}/{len(manager_links)}: {manager_info['url']}")
                holdings = self.scrape_manager(manager_info['url'], manager_info.get('name', ''))
                all_holdings.extend(holdings)
                print(f"  Found {len(holdings)} holdings")
                
                time.sleep(0.5)
            except Exception as e:
                print(f"Error scraping {manager_info['url']}: {e}")
        
        print(f"\nTotal holdings scraped: {len(all_holdings)}")
        
        # Get unique tickers for stock data enrichment
        unique_tickers = list(set(h['ticker'] for h in all_holdings if h.get('ticker')))
        print(f"Found {len(unique_tickers)} unique stocks")
        
        # Second pass: Enrich with detailed stock data
        print("\nEnriching holdings with detailed stock data...")
        for i, ticker in enumerate(unique_tickers):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(unique_tickers)} stocks processed")
            
            stock_data = self.scrape_stock_page(ticker)
            
            # Update all holdings for this ticker
            for holding in all_holdings:
                if holding.get('ticker') == ticker:
                    # Add stock data to holding
                    holding['market_cap'] = stock_data.get('market_cap', 0)
                    holding['sector'] = stock_data.get('sector', 'Unknown')
                    holding['industry'] = stock_data.get('industry', 'Unknown')
                    holding['current_price'] = stock_data.get('current_price', holding.get('current_price', 0))
                    holding['52_week_high'] = stock_data.get('52_week_high', 0)
                    holding['52_week_low'] = stock_data.get('52_week_low', 0)
                    holding['pe_ratio'] = stock_data.get('pe_ratio', 0)
                    holding['dividend_yield'] = stock_data.get('dividend_yield', 0)
        
        # Save the cache
        self.save_stock_cache()
        print(f"\nStock cache updated with {len(self.stock_cache)} stocks")
        
        # Save holdings
        self.save_holdings(all_holdings)
        
        return all_holdings
    
    def scrape_manager(self, manager_link, hint_name=""):
        """Scrape holdings for a specific manager"""
        # Handle URLs
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
        
        # Extract manager name
        manager_name = hint_name
        h1_elem = soup.find('h1')
        if h1_elem and h1_elem.text.strip():
            manager_name = h1_elem.text.strip()
        
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
        
        # Find the holdings table
        table = soup.find('table', {'id': 'grid'}) or soup.find('table')
        
        if table:
            rows = table.find_all('tr')
            
            # Find header row to understand column positions
            header_row = None
            for row in rows:
                if row.find('th'):
                    header_row = row
                    break
            
            # Map column positions
            col_map = {}
            if header_row:
                headers = [th.text.strip().lower() for th in header_row.find_all('th')]
                for i, header in enumerate(headers):
                    if 'stock' in header or 'company' in header:
                        col_map['stock'] = i
                    elif 'portfolio' in header and '%' in header:
                        col_map['percent'] = i
                    elif 'shares' in header:
                        col_map['shares'] = i
                    elif 'activity' in header:
                        col_map['activity'] = i
                    elif 'reported' in header and 'price' in header:
                        col_map['reported_price'] = i
                    elif 'value' in header:
                        col_map['value'] = i
            
            for row in rows:
                # Skip header rows
                if row.find('th') or not row.find('td'):
                    continue
                
                cells = row.find_all('td')
                
                if len(cells) >= 2:
                    # Extract stock info
                    stock_cell = cells[col_map.get('stock', 0)]
                    stock_link = stock_cell.find('a')
                    
                    if stock_link and 'stock.php?sym=' in stock_link.get('href', ''):
                        ticker = stock_link.get('href').split('sym=')[-1]
                        stock_name = stock_link.text.strip()
                        
                        # Clean stock name
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
                            'current_price': None,
                            'value': None
                        }
                        
                        # Extract data based on column mapping
                        if 'percent' in col_map and col_map['percent'] < len(cells):
                            percent_text = cells[col_map['percent']].text.strip()
                            holding['portfolio_percent'] = percent_text.replace('%', '').strip()
                        
                        if 'shares' in col_map and col_map['shares'] < len(cells):
                            holding['shares'] = cells[col_map['shares']].text.strip()
                        
                        if 'activity' in col_map and col_map['activity'] < len(cells):
                            holding['recent_activity'] = cells[col_map['activity']].text.strip()
                        
                        if 'reported_price' in col_map and col_map['reported_price'] < len(cells):
                            price = self.clean_price(cells[col_map['reported_price']].text.strip())
                            if price:
                                holding['reported_price'] = price
                        
                        if 'value' in col_map and col_map['value'] < len(cells):
                            holding['value'] = cells[col_map['value']].text.strip()
                        
                        holdings.append(holding)
        
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
        
        if not all_holdings:
            print("No holdings found to save")
            return
        
        # Save daily snapshot
        daily_filename = os.path.join(self.holdings_dir, f'holdings_{today}.csv')
        df = pd.DataFrame(all_holdings)
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['portfolio_percent', 'reported_price', 'current_price', 
                          '52_week_high', '52_week_low', 'market_cap', 
                          'pe_ratio', 'dividend_yield']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.to_csv(daily_filename, index=False)
        
        # Save as latest
        df.to_csv('latest_holdings.csv', index=False)
        
        # Create metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_holdings': len(all_holdings),
            'managers_tracked': len(df['manager'].unique()) if 'manager' in df.columns else 0,
            'unique_stocks': len(df['ticker'].unique()) if 'ticker' in df.columns else 0,
            'holdings_with_prices': len(df[df['current_price'].notna()]) if 'current_price' in df.columns else 0,
            'holdings_with_market_cap': len(df[df['market_cap'] > 0]) if 'market_cap' in df.columns else 0,
            'holdings_with_sector': len(df[df['sector'] != 'Unknown']) if 'sector' in df.columns else 0,
            'file_location': daily_filename
        }
        
        with open('scrape_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nScraping Summary:")
        print(f"- Total holdings: {metadata['total_holdings']}")
        print(f"- Managers tracked: {metadata['managers_tracked']}")
        print(f"- Unique stocks: {metadata['unique_stocks']}")
        print(f"- Holdings with prices: {metadata['holdings_with_prices']}")
        print(f"- Holdings with market cap: {metadata['holdings_with_market_cap']}")
        print(f"- Holdings with sector: {metadata['holdings_with_sector']}")
        print(f"- Saved to: {daily_filename}")

def main():
    scraper = DataromaScraper()
    
    # Test connectivity
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