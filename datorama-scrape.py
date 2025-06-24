import requests
from bs4 import BeautifulSoup
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
    
    def should_refresh_stock(self, ticker):
        """Check if stock data should be refreshed (monthly update)"""
        if ticker not in self.stock_cache:
            return True
        
        cached_data = self.stock_cache[ticker]
        if 'last_updated' in cached_data:
            last_updated = datetime.fromisoformat(cached_data['last_updated'])
            # Update monthly instead of weekly
            if datetime.now() - last_updated > timedelta(days=30):
                return True
        
        # Check if we have comprehensive data
        required_fields = ['market_cap', 'sector', 'current_price', '52_week_high', '52_week_low']
        if any(field not in cached_data or not cached_data.get(field) for field in required_fields):
            return True
        
        return False
    
    def scrape_stock_page(self, ticker):
        """Enhanced stock page scraping based on actual Dataroma structure"""
        # Check if we need to refresh
        if not self.should_refresh_stock(ticker):
            return self.stock_cache[ticker]
        
        stock_url = f"{self.base_url}stock.php?sym={ticker}"
        print(f"    Fetching stock data for {ticker}")
        
        try:
            response = requests.get(stock_url, timeout=10, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            stock_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat()
            }
            
            # Extract company name from h1
            h1 = soup.find('h1')
            if h1:
                company_text = h1.text.strip()
                # Remove ticker symbol in parentheses
                company_name = re.sub(r'\s*\([A-Z]+\)\s*$', '', company_text).strip()
                if company_name:
                    stock_data['company_name'] = company_name
            
            # Extract data from table - Dataroma uses simple table layout
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].text.strip().lower()
                        value = cells[1].text.strip()
                        
                        # Extract sector
                        if 'sector' in label:
                            stock_data['sector'] = value if value else 'Unknown'
                        
                        # Extract ownership stats
                        elif 'ownership count' in label:
                            try:
                                stock_data['ownership_count'] = int(value)
                            except:
                                pass
                        
                        elif 'ownership rank' in label:
                            try:
                                stock_data['ownership_rank'] = int(value)
                            except:
                                pass
                        
                        elif '% of all portfolios' in label:
                            try:
                                stock_data['pct_of_all_portfolios'] = float(value.replace('%', ''))
                            except:
                                pass
                        
                        elif 'hold price' in label:
                            price = self.clean_price(value)
                            if price:
                                stock_data['hold_price'] = price
            
            # Try to fetch additional data from external sources or page content
            # Look for current price in page (Dataroma might have it in JavaScript or elsewhere)
            page_text = soup.get_text()
            
            # Look for price patterns
            price_patterns = [
                r'Current Price[:\s]+\$?([\d,]+\.?\d*)',
                r'Price[:\s]+\$?([\d,]+\.?\d*)',
                r'\$?([\d,]+\.?\d*)\s+(?:USD|usd)',
            ]
            
            for pattern in price_patterns:
                match = re.search(pattern, page_text)
                if match:
                    price = self.clean_price(match.group(1))
                    if price and 0.01 <= price <= 100000:
                        stock_data['current_price'] = price
                        break
            
            # Use hold price as current price if we don't have it
            if 'current_price' not in stock_data and 'hold_price' in stock_data:
                stock_data['current_price'] = stock_data['hold_price']
            
            # Try to extract market cap, PE, etc. from the page
            # These might not be on Dataroma, but we'll try
            market_cap_patterns = [
                r'Market\s*Cap[:\s]+\$?([\d,]+\.?\d*)\s*([BMKT])',
                r'Mkt\s*Cap[:\s]+\$?([\d,]+\.?\d*)\s*([BMKT])',
            ]
            
            for pattern in market_cap_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1).replace(',', ''))
                    multiplier = match.group(2).upper()
                    if multiplier == 'B':
                        value *= 1_000_000_000
                    elif multiplier == 'M':
                        value *= 1_000_000
                    elif multiplier == 'K':
                        value *= 1_000
                    elif multiplier == 'T':
                        value *= 1_000_000_000_000
                    stock_data['market_cap'] = value
                    break
            
            # Extract P/E ratio
            pe_patterns = [
                r'P/E\s*(?:Ratio)?[:\s]+([\d,]+\.?\d*)',
                r'PE\s*(?:Ratio)?[:\s]+([\d,]+\.?\d*)',
            ]
            
            for pattern in pe_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    pe = self.clean_price(match.group(1))
                    if pe and 0 < pe < 1000:
                        stock_data['pe_ratio'] = pe
                        break
            
            # Extract 52-week range
            range_pattern = r'52\s*Week\s*Range[:\s]+([\d.]+)\s*-\s*([\d.]+)'
            range_match = re.search(range_pattern, page_text, re.IGNORECASE)
            if range_match:
                low = self.clean_price(range_match.group(1))
                high = self.clean_price(range_match.group(2))
                if low and high:
                    stock_data['52_week_low'] = low
                    stock_data['52_week_high'] = high
            
            # Fill in defaults
            if 'sector' not in stock_data:
                stock_data['sector'] = 'Unknown'
            if 'industry' not in stock_data:
                stock_data['industry'] = 'Unknown'
            
            # Update cache
            self.stock_cache[ticker] = stock_data
            
            # Small delay to be polite
            time.sleep(0.5)
            
            return stock_data
            
        except Exception as e:
            print(f"    Error fetching stock page for {ticker}: {e}")
            # Return cached data if available, otherwise minimal data
            return self.stock_cache.get(ticker, {
                'ticker': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown'
            })
    
    def clean_price(self, price_str):
        """Clean price string to float"""
        if not price_str:
            return None
        try:
            # Remove common non-numeric characters
            cleaned = price_str.replace('$', '').replace(',', '').replace(' ', '').strip()
            
            # Handle special cases
            if cleaned in ['-', 'N/A', 'n/a', '']:
                return None
                
            # Extract numeric value
            match = re.search(r'([\d.]+)', cleaned)
            if match:
                return float(match.group(1))
                
        except (ValueError, AttributeError) as e:
            pass
        return None
    
    def scrape_dataroma(self):
        """Main scraping function - workflow ready"""
        print("="*60)
        print("DATAROMA SCRAPER - WORKFLOW MODE")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get list of managers
        managers_url = self.base_url + "managers.php"
        print(f"\nFetching managers from: {managers_url}")
        
        try:
            response = requests.get(managers_url, timeout=30, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching managers page: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        all_holdings = []
        manager_links = []
        
        # Extract manager links - ONLY get holdings.php?m= links
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Only accept holdings.php?m= pattern (not stock.php)
            if 'holdings.php?m=' in href and 'stock.php' not in href:
                manager_info = {
                    'url': href,
                    'name': link.text.strip() if link.text else ""
                }
                # Avoid duplicates
                if not any(m['url'] == manager_info['url'] for m in manager_links):
                    manager_links.append(manager_info)
        
        print(f"Found {len(manager_links)} managers to scrape")
        
        # Process all managers
        successful_managers = 0
        failed_managers = 0
        
        for idx, manager_info in enumerate(manager_links, 1):
            try:
                print(f"\n[{idx}/{len(manager_links)}] Scraping: {manager_info['url']}")
                holdings = self.scrape_manager(manager_info['url'], manager_info.get('name', ''))
                
                if holdings:
                    all_holdings.extend(holdings)
                    successful_managers += 1
                    print(f"  ✓ Found {len(holdings)} holdings")
                else:
                    failed_managers += 1
                    print(f"  ✗ No holdings found")
                
                # Rate limiting
                time.sleep(1)
                
                # Longer pause every 10 managers
                if idx % 10 == 0:
                    print(f"  [Rate limit pause - 5 seconds]")
                    time.sleep(5)
                    
            except Exception as e:
                failed_managers += 1
                print(f"  ✗ Error: {e}")
        
        print(f"\n{'='*60}")
        print(f"MANAGER SCRAPING COMPLETE")
        print(f"Successful: {successful_managers}")
        print(f"Failed: {failed_managers}")
        print(f"Total holdings: {len(all_holdings)}")
        
        # Get unique tickers for stock data enrichment
        unique_tickers = list(set(h['ticker'] for h in all_holdings if h.get('ticker')))
        print(f"\nUnique stocks found: {len(unique_tickers)}")
        
        # Second pass: Enrich with detailed stock data
        if unique_tickers:
            print(f"\n{'='*60}")
            print("ENRICHING HOLDINGS WITH STOCK DATA")
            print(f"{'='*60}")
            
            enriched_count = 0
            for i, ticker in enumerate(unique_tickers):
                if i % 25 == 0:
                    print(f"\nProgress: {i}/{len(unique_tickers)} stocks processed")
                
                stock_data = self.scrape_stock_page(ticker)
                
                # Update all holdings for this ticker
                for holding in all_holdings:
                    if holding.get('ticker') == ticker:
                        # Add all available stock data
                        for key, value in stock_data.items():
                            if key not in ['ticker', 'last_updated']:
                                holding[key] = value
                        
                        # Ensure we have market_cap as a number
                        if 'market_cap' not in holding:
                            holding['market_cap'] = 0
                        
                        if stock_data.get('current_price', 0) > 0:
                            enriched_count += 1
            
            print(f"\nEnriched {enriched_count} holdings with price data")
        
        # Save the cache
        self.save_stock_cache()
        print(f"\nStock cache updated with {len(self.stock_cache)} stocks")
        
        # Save holdings
        self.save_holdings(all_holdings)
        
        print(f"\n{'='*60}")
        print("SCRAPING COMPLETE")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        return all_holdings
    
    def scrape_manager(self, manager_link, hint_name=""):
        """Scrape holdings for a specific manager based on actual page structure"""
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
            print(f"  Error fetching: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holdings = []
        
        # Extract manager name
        manager_name = hint_name
        h1_elem = soup.find('h1')
        if h1_elem and h1_elem.text.strip():
            manager_name = h1_elem.text.strip()
        
        # Extract portfolio date
        portfolio_date = datetime.now().strftime('%Y-%m-%d')
        date_text = soup.get_text()
        
        # Look for various date patterns
        date_patterns = [
            r'Portfolio\s*date:\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'Period:\s*Q\d\s+\d{4}.*?Portfolio\s*date:\s*(\d{1,2}\s+\w+\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    date_str = match.group(1)
                    portfolio_date = datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
                    break
                except:
                    pass
        
        # Find the holdings table
        holdings_table = None
        
        # Look for table with id='grid' (common pattern on Dataroma)
        holdings_table = soup.find('table', {'id': 'grid'})
        
        if not holdings_table:
            # Look for table containing stock links
            tables = soup.find_all('table')
            for table in tables:
                if table.find('a', href=lambda x: x and 'stock.php' in x):
                    holdings_table = table
                    break
        
        if holdings_table:
            rows = holdings_table.find_all('tr')
            
            # Find header row and map columns
            headers = []
            col_map = {}
            header_row_idx = -1
            
            for idx, row in enumerate(rows):
                ths = row.find_all('th')
                if ths:
                    header_row_idx = idx
                    # Extract header text, handling multi-line headers
                    for i, th in enumerate(ths):
                        # Get all text and join with space
                        header_text = ' '.join(th.stripped_strings).lower()
                        headers.append(header_text)
                        
                        # Map columns based on header content
                        if any(word in header_text for word in ['stock', 'company', 'holding']):
                            col_map['stock'] = i
                        elif '% of portfolio' in header_text or 'portfolio' in header_text and '%' in header_text:
                            col_map['percent'] = i
                        elif 'shares' in header_text:
                            col_map['shares'] = i
                        elif 'recent activity' in header_text or 'activity' in header_text:
                            col_map['activity'] = i
                        elif 'reported price' in header_text:
                            col_map['reported_price'] = i
                        elif 'current price' in header_text:
                            col_map['current_price'] = i
                        elif 'value' in header_text:
                            col_map['value'] = i
                    break
            
            # Process data rows
            for row_idx, row in enumerate(rows):
                # Skip header row
                if row_idx <= header_row_idx:
                    continue
                
                cells = row.find_all('td')
                if not cells or len(cells) < 2:
                    continue
                
                # Look for stock link
                stock_link = None
                ticker = None
                stock_name = None
                
                # Try mapped column first
                if 'stock' in col_map and col_map['stock'] < len(cells):
                    cell = cells[col_map['stock']]
                    stock_link = cell.find('a', href=lambda x: x and 'stock.php' in x)
                
                # Search all cells if not found
                if not stock_link:
                    for cell in cells:
                        link = cell.find('a', href=lambda x: x and 'stock.php' in x)
                        if link:
                            stock_link = link
                            break
                
                if stock_link:
                    # Extract ticker from URL
                    href = stock_link.get('href', '')
                    ticker_match = re.search(r'[?&]sym=([A-Z0-9.-]+)', href, re.IGNORECASE)
                    if ticker_match:
                        ticker = ticker_match.group(1).upper()
                        stock_name = stock_link.text.strip()
                        
                        # Clean stock name (remove ticker if present)
                        stock_name = re.sub(r'^[A-Z0-9.-]+\s*-\s*', '', stock_name).strip()
                        
                        holding = {
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'portfolio_date': portfolio_date,
                            'manager': manager_name,
                            'stock': stock_name,
                            'ticker': ticker,
                            'portfolio_percent': "0",
                            'shares': "0",
                            'recent_activity': "",
                            'reported_price': 0,
                            'current_price': 0,
                            'value': 0
                        }
                        
                        # Extract percentage
                        if 'percent' in col_map and col_map['percent'] < len(cells):
                            percent_text = cells[col_map['percent']].text.strip()
                            percent_val = self.extract_percentage(percent_text)
                            if percent_val is not None:
                                holding['portfolio_percent'] = str(percent_val)
                        
                        # Extract shares
                        if 'shares' in col_map and col_map['shares'] < len(cells):
                            shares_text = cells[col_map['shares']].text.strip()
                            shares_val = self.extract_number(shares_text)
                            if shares_val:
                                holding['shares'] = shares_val
                        
                        # Extract activity
                        if 'activity' in col_map and col_map['activity'] < len(cells):
                            activity_text = cells[col_map['activity']].text.strip()
                            if activity_text and activity_text != '-':
                                holding['recent_activity'] = activity_text
                        
                        # Extract reported price
                        if 'reported_price' in col_map and col_map['reported_price'] < len(cells):
                            price_text = cells[col_map['reported_price']].text.strip()
                            price = self.clean_price(price_text)
                            if price:
                                holding['reported_price'] = price
                        
                        # Extract current price (if available on page)
                        if 'current_price' in col_map and col_map['current_price'] < len(cells):
                            price_text = cells[col_map['current_price']].text.strip()
                            price = self.clean_price(price_text)
                            if price:
                                holding['current_price'] = price
                        
                        # Extract value
                        if 'value' in col_map and col_map['value'] < len(cells):
                            value_text = cells[col_map['value']].text.strip()
                            # Convert value (might be in millions)
                            value = self.extract_value(value_text)
                            if value:
                                holding['value'] = value
                        
                        holdings.append(holding)
        
        return holdings
    
    def extract_percentage(self, text):
        """Extract percentage value from text"""
        if not text:
            return None
            
        # Clean the text
        text = text.strip().replace('*', '').replace(',', '.')
        
        # Try to find percentage pattern
        patterns = [
            r'(\d+\.?\d*)\s*%?',      # 12.5 or 12.5%
            r'^(\d+\.?\d*)',           # Just a number at start
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 100:  # Sanity check
                        return val
                except:
                    pass
        
        return None
    
    def extract_number(self, text):
        """Extract number from text (for shares)"""
        if not text:
            return "0"
            
        # Remove commas and spaces
        text = text.replace(',', '').replace(' ', '').strip()
        
        # Check if it's a valid number
        if re.match(r'^\d+$', text):
            return text
            
        # Try to extract number
        match = re.search(r'(\d+)', text)
        if match:
            return match.group(1)
            
        return "0"
    
    def extract_value(self, text):
        """Extract value which might be in thousands or millions"""
        if not text:
            return 0
            
        # Remove $ and commas
        text = text.replace('$', '').replace(',', '').strip()
        
        try:
            # Check if it's in millions (M) or thousands (K)
            if text.endswith('M'):
                return float(text[:-1]) * 1_000_000
            elif text.endswith('K'):
                return float(text[:-1]) * 1_000
            else:
                # Assume it's already in dollars
                return float(text)
        except:
            return 0
    
    def save_holdings(self, all_holdings):
        """Save holdings with enhanced metadata"""
        today = datetime.now().strftime('%Y%m%d')
        
        if not all_holdings:
            print("\nNo holdings found to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_holdings)
        
        # Ensure numeric columns are properly formatted
        numeric_columns = {
            'portfolio_percent': 0,
            'reported_price': 0,
            'current_price': 0,
            'market_cap': 0,
            'pe_ratio': 0,
            'dividend_yield': 0,
            '52_week_high': 0,
            '52_week_low': 0,
            'ownership_count': 0,
            'ownership_rank': 0,
            'pct_of_all_portfolios': 0,
            'value': 0
        }
        
        for col, default_val in numeric_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
        
        # Save files
        daily_filename = os.path.join(self.holdings_dir, f'holdings_{today}.csv')
        df.to_csv(daily_filename, index=False)
        df.to_csv('latest_holdings.csv', index=False)
        
        # Create enhanced metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'scrape_date': today,
            'total_holdings': len(all_holdings),
            'managers_tracked': len(df['manager'].unique()) if 'manager' in df.columns else 0,
            'unique_stocks': len(df['ticker'].unique()) if 'ticker' in df.columns else 0,
            'holdings_with_portfolio_pct': len(df[df['portfolio_percent'] > 0]) if 'portfolio_percent' in df.columns else 0,
            'holdings_with_prices': len(df[df['current_price'] > 0]) if 'current_price' in df.columns else 0,
            'holdings_with_reported_prices': len(df[df['reported_price'] > 0]) if 'reported_price' in df.columns else 0,
            'holdings_with_market_cap': len(df[df['market_cap'] > 0]) if 'market_cap' in df.columns else 0,
            'holdings_with_sector': len(df[(df['sector'] != 'Unknown') & df['sector'].notna()]) if 'sector' in df.columns else 0,
            'file_location': daily_filename,
            'data_quality': {
                'pct_with_portfolio_percent': round(len(df[df['portfolio_percent'] > 0]) / len(df) * 100, 1) if len(df) > 0 else 0,
                'pct_with_prices': round(len(df[df['current_price'] > 0]) / len(df) * 100, 1) if len(df) > 0 else 0,
                'pct_with_market_cap': round(len(df[df['market_cap'] > 0]) / len(df) * 100, 1) if len(df) > 0 else 0,
                'pct_with_sector': round(len(df[(df['sector'] != 'Unknown') & df['sector'].notna()]) / len(df) * 100, 1) if len(df) > 0 else 0
            },
            'cache_info': {
                'total_cached_stocks': len(self.stock_cache),
                'cache_update_frequency': 'monthly'
            }
        }
        
        with open('scrape_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print("SCRAPING SUMMARY")
        print(f"{'='*60}")
        print(f"Total holdings: {metadata['total_holdings']:,}")
        print(f"Managers tracked: {metadata['managers_tracked']}")
        print(f"Unique stocks: {metadata['unique_stocks']}")
        print(f"\nData Quality:")
        print(f"- With portfolio %: {metadata['holdings_with_portfolio_pct']:,} ({metadata['data_quality']['pct_with_portfolio_percent']}%)")
        print(f"- With current price: {metadata['holdings_with_prices']:,} ({metadata['data_quality']['pct_with_prices']}%)")
        print(f"- With reported price: {metadata['holdings_with_reported_prices']:,}")
        print(f"- With market cap: {metadata['holdings_with_market_cap']:,} ({metadata['data_quality']['pct_with_market_cap']}%)")
        print(f"- With sector: {metadata['holdings_with_sector']:,} ({metadata['data_quality']['pct_with_sector']}%)")
        print(f"\nFiles saved:")
        print(f"- {daily_filename}")
        print(f"- latest_holdings.csv")
        print(f"- scrape_metadata.json")

def main():
    """Main function - workflow ready, no interaction needed"""
    scraper = DataromaScraper()
    
    # Test connectivity
    print("Testing Dataroma connectivity...")
    test_url = "https://www.dataroma.com/m/home.php"
    try:
        response = requests.get(test_url, timeout=10, headers=scraper.headers)
        if response.status_code == 200:
            print(f"✓ Successfully connected to Dataroma (status: {response.status_code})")
        else:
            print(f"⚠ Warning: Dataroma returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Error connecting to Dataroma: {e}")
        return 1
    
    # Run the scraper
    try:
        scraper.scrape_dataroma()
        return 0
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())