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
        self.debug_dir = "debug"
        self.ensure_directories()
        self.stock_cache = self.load_stock_cache()
        
        # Set headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.holdings_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs("analysis", exist_ok=True)
        os.makedirs("debug", exist_ok=True)
    
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
    
    def debug_save_html(self, html_content, filename):
        """Save HTML content for debugging"""
        debug_file = os.path.join(self.debug_dir, filename)
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  DEBUG: Saved HTML to {debug_file}")
    
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
            
            # Current Price
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
            
            # Update cache
            self.stock_cache[ticker] = stock_data
            
            # Small delay to be polite
            time.sleep(0.3)
            
            return stock_data
            
        except Exception as e:
            print(f"    Error fetching stock page for {ticker}: {e}")
            return self.stock_cache.get(ticker, {'ticker': ticker})
    
    def parse_market_cap(self, cap_str):
        """Parse market cap string to number"""
        try:
            cap_str = cap_str.replace('$', '').replace(' ', '').strip()
            
            if 'T' in cap_str or 'trillion' in cap_str.lower():
                return float(re.search(r'([\d.]+)', cap_str).group(1)) * 1_000_000_000_000
            elif 'B' in cap_str or 'billion' in cap_str.lower():
                return float(re.search(r'([\d.]+)', cap_str).group(1)) * 1_000_000_000
            elif 'M' in cap_str or 'million' in cap_str.lower():
                return float(re.search(r'([\d.]+)', cap_str).group(1)) * 1_000_000
            else:
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
        
        # Save first page for debugging
        self.debug_save_html(response.text, "managers_page.html")
        
        all_holdings = []
        manager_links = []
        
        # Extract manager links - try multiple patterns
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Look for various patterns that might indicate a manager page
            if any(pattern in href for pattern in ['holdings.php?m=', 'holdings.php', 'm=']):
                manager_info = {
                    'url': href,
                    'name': link.text.strip() if link.text else ""
                }
                manager_links.append(manager_info)
        
        print(f"Found {len(manager_links)} managers to scrape")
        
        # Debug: print first few manager links
        if manager_links:
            print("\nFirst 3 manager links found:")
            for ml in manager_links[:3]:
                print(f"  - {ml['url']} ({ml['name']})")
        
        # First pass: Get all holdings (limit to first 3 for debugging)
        for idx, manager_info in enumerate(manager_links[:3], 1):  # LIMIT TO 3 FOR DEBUGGING
            try:
                print(f"\nScraping manager {idx}/3 (DEBUG MODE): {manager_info['url']}")
                holdings = self.scrape_manager(manager_info['url'], manager_info.get('name', ''), debug_index=idx)
                all_holdings.extend(holdings)
                print(f"  Found {len(holdings)} holdings")
                
                time.sleep(1)  # Increased delay
            except Exception as e:
                print(f"Error scraping {manager_info['url']}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nTotal holdings scraped: {len(all_holdings)}")
        
        # Get unique tickers for stock data enrichment
        unique_tickers = list(set(h['ticker'] for h in all_holdings if h.get('ticker')))
        print(f"Found {len(unique_tickers)} unique stocks")
        
        # Second pass: Enrich with detailed stock data
        if unique_tickers:
            print("\nEnriching holdings with detailed stock data...")
            for i, ticker in enumerate(unique_tickers[:10]):  # Limit for debugging
                if i % 5 == 0:
                    print(f"Progress: {i}/{min(len(unique_tickers), 10)} stocks processed")
                
                stock_data = self.scrape_stock_page(ticker)
                
                # Update all holdings for this ticker
                for holding in all_holdings:
                    if holding.get('ticker') == ticker:
                        holding['market_cap'] = stock_data.get('market_cap', 0)
                        holding['sector'] = stock_data.get('sector', 'Unknown')
                        holding['industry'] = stock_data.get('industry', 'Unknown')
                        holding['current_price'] = stock_data.get('current_price', holding.get('current_price', 0))
        
        # Save the cache
        self.save_stock_cache()
        print(f"\nStock cache updated with {len(self.stock_cache)} stocks")
        
        # Save holdings
        self.save_holdings(all_holdings)
        
        return all_holdings
    
    def scrape_manager(self, manager_link, hint_name="", debug_index=0):
        """Scrape holdings for a specific manager with better debugging"""
        # Handle URLs
        if manager_link.startswith('/'):
            manager_url = "https://www.dataroma.com" + manager_link
        elif not manager_link.startswith('http'):
            manager_url = self.base_url + manager_link.lstrip('/')
        else:
            manager_url = manager_link
        
        print(f"  Full URL: {manager_url}")
        
        try:
            response = requests.get(manager_url, timeout=30, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {manager_url}: {e}")
            return []
        
        # Save HTML for debugging
        if debug_index <= 3:  # Save first 3 for debugging
            self.debug_save_html(response.text, f"manager_{debug_index}_holdings.html")
        
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
        date_text = soup.get_text()
        date_patterns = [
            r'Portfolio date:\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'As of\s+(\d{1,2}\s+\w+\s+\d{4})',
            r'Updated:\s*(\d{1,2}\s+\w+\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    portfolio_date = datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
                    print(f"  Portfolio date: {portfolio_date}")
                    break
                except:
                    pass
        
        # Debug: print all tables found
        tables = soup.find_all('table')
        print(f"  DEBUG: Found {len(tables)} tables on page")
        
        # Try to find the holdings table with various methods
        holdings_table = None
        
        # Method 1: Look for table with id='grid'
        holdings_table = soup.find('table', {'id': 'grid'})
        if holdings_table:
            print("  DEBUG: Found table with id='grid'")
        
        # Method 2: Look for table with class containing 'holdings' or 'portfolio'
        if not holdings_table:
            for table in tables:
                table_class = table.get('class', [])
                if isinstance(table_class, list):
                    table_class = ' '.join(table_class)
                else:
                    table_class = str(table_class)
                
                if any(keyword in table_class.lower() for keyword in ['holding', 'portfolio', 'grid']):
                    holdings_table = table
                    print(f"  DEBUG: Found table with class: {table_class}")
                    break
        
        # Method 3: Look for table containing stock links
        if not holdings_table:
            for table in tables:
                if table.find('a', href=lambda x: x and 'stock.php' in x):
                    holdings_table = table
                    print("  DEBUG: Found table containing stock.php links")
                    break
        
        # Method 4: Just use the largest table
        if not holdings_table and tables:
            holdings_table = max(tables, key=lambda t: len(t.find_all('tr')))
            print(f"  DEBUG: Using largest table with {len(holdings_table.find_all('tr'))} rows")
        
        if holdings_table:
            rows = holdings_table.find_all('tr')
            print(f"  DEBUG: Processing {len(rows)} rows")
            
            # Find header row to understand column positions
            header_row = None
            header_index = 0
            for i, row in enumerate(rows):
                if row.find('th'):
                    header_row = row
                    header_index = i
                    break
            
            # Map column positions
            col_map = {}
            if header_row:
                headers = [th.text.strip().lower() for th in header_row.find_all('th')]
                print(f"  DEBUG: Headers found: {headers}")
                
                for i, header in enumerate(headers):
                    if any(keyword in header for keyword in ['stock', 'company', 'name', 'holding']):
                        col_map['stock'] = i
                    elif 'portfolio' in header and '%' in header:
                        col_map['percent'] = i
                    elif '%' in header and 'percent' not in col_map:
                        col_map['percent'] = i
                    elif 'shares' in header:
                        col_map['shares'] = i
                    elif 'activity' in header:
                        col_map['activity'] = i
                    elif 'reported' in header and 'price' in header:
                        col_map['reported_price'] = i
                    elif 'value' in header:
                        col_map['value'] = i
            
            print(f"  DEBUG: Column mapping: {col_map}")
            
            # Process data rows
            data_rows_processed = 0
            for row_idx, row in enumerate(rows):
                # Skip header rows
                if row.find('th') or not row.find('td'):
                    continue
                
                cells = row.find_all('td')
                
                if len(cells) >= 2:
                    # Try to find stock link in any cell
                    stock_link = None
                    stock_cell_idx = col_map.get('stock', 0)
                    
                    # First try the mapped column
                    if stock_cell_idx < len(cells):
                        stock_link = cells[stock_cell_idx].find('a', href=lambda x: x and 'stock.php' in x)
                    
                    # If not found, search all cells
                    if not stock_link:
                        for cell in cells:
                            stock_link = cell.find('a', href=lambda x: x and 'stock.php' in x)
                            if stock_link:
                                break
                    
                    if stock_link:
                        href = stock_link.get('href', '')
                        ticker_match = re.search(r'[?&]sym=([A-Z0-9.-]+)', href, re.IGNORECASE)
                        
                        if ticker_match:
                            ticker = ticker_match.group(1).upper()
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
                            
                            # Extract data based on column mapping or search cells
                            # Portfolio percent
                            if 'percent' in col_map and col_map['percent'] < len(cells):
                                percent_text = cells[col_map['percent']].text.strip()
                                percent_text = percent_text.replace('%', '').strip()
                                if percent_text and percent_text != '-':
                                    holding['portfolio_percent'] = percent_text
                            
                            # If no percent found, look for % in any cell
                            if holding['portfolio_percent'] == "0":
                                for cell in cells:
                                    text = cell.text.strip()
                                    if '%' in text and text.replace('%', '').replace('.', '').replace(',', '').isdigit():
                                        holding['portfolio_percent'] = text.replace('%', '').strip()
                                        break
                            
                            # Shares
                            if 'shares' in col_map and col_map['shares'] < len(cells):
                                shares_text = cells[col_map['shares']].text.strip().replace(',', '')
                                if shares_text and shares_text != '-':
                                    holding['shares'] = shares_text
                            
                            # Activity
                            if 'activity' in col_map and col_map['activity'] < len(cells):
                                holding['recent_activity'] = cells[col_map['activity']].text.strip()
                            
                            # Price
                            if 'reported_price' in col_map and col_map['reported_price'] < len(cells):
                                price = self.clean_price(cells[col_map['reported_price']].text.strip())
                                if price:
                                    holding['reported_price'] = price
                            
                            holdings.append(holding)
                            data_rows_processed += 1
                            
                            if data_rows_processed <= 3:  # Print first 3 for debugging
                                print(f"    DEBUG: Found holding - {ticker}: {stock_name} ({holding['portfolio_percent']}%)")
            
            print(f"  DEBUG: Processed {data_rows_processed} data rows")
        else:
            print("  DEBUG: No holdings table found!")
            # Debug: print some page content
            page_text = soup.get_text()[:500]
            print(f"  DEBUG: Page preview: {page_text}")
        
        return holdings
    
    def clean_price(self, price_str):
        """Clean price string to float"""
        if not price_str:
            return None
        try:
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
        headers = scraper.headers
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
    
    print("\n" + "="*50)
    print("DEBUG: Check the 'debug' folder for saved HTML files")
    print("This will help you see what the scraper is actually receiving")
    print("="*50)

if __name__ == "__main__":
    main()