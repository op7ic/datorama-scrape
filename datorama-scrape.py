import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import os
import pandas as pd
import re
import time
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImprovedDataromaScraper:
    def __init__(self):
        self.base_url = "https://www.dataroma.com/m/"
        self.holdings_dir = "holdings"
        self.cache_dir = "cache"
        self.debug_dir = "debug"
        self.ensure_directories()
        self.stock_cache = self.load_cache('stock_data_cache.json')
        self.price_cache = self.load_cache('price_cache.json')
        
        # Browser headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def ensure_directories(self):
        """Create necessary directories"""
        for dir_name in [self.holdings_dir, self.cache_dir, self.debug_dir, "analysis"]:
            os.makedirs(dir_name, exist_ok=True)
    
    def load_cache(self, filename):
        """Load cache file"""
        cache_file = os.path.join(self.cache_dir, filename)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self, data, filename):
        """Save cache file"""
        cache_file = os.path.join(self.cache_dir, filename)
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def debug_save_html(self, html_content, filename):
        """Save HTML for debugging"""
        debug_file = os.path.join(self.debug_dir, filename)
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def fetch_page(self, url, timeout=30):
        """Fetch a page with retry logic"""
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def clean_text(self, text):
        """Clean text by removing extra whitespace"""
        if not text:
            return ""
        return ' '.join(text.split()).strip()
    
    def extract_percentage(self, text):
        """Extract percentage from various formats"""
        if not text:
            return 0.0
        
        # Clean the text
        text = self.clean_text(text).replace('*', '').replace(',', '.')
        
        # Try various patterns
        patterns = [
            r'(\d+\.?\d*)\s*%',     # 12.5% or 12%
            r'^(\d+\.?\d*)$',       # Just a number
            r'(\d+\.?\d*)\s*$',     # Number at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 100:
                        return val
                except:
                    pass
        
        return 0.0
    
    def extract_number(self, text):
        """Extract number from text (for shares)"""
        if not text:
            return 0
        
        # Remove formatting
        text = str(text).replace(',', '').replace(' ', '').replace('$', '').strip()
        
        # Try to extract number
        match = re.search(r'(\d+)', text)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
        
        return 0
    
    def extract_price(self, text):
        """Extract price from text"""
        if not text:
            return 0.0
        
        # Clean text
        text = str(text).replace('$', '').replace(',', '').strip()
        
        # Handle special cases
        if text in ['-', 'N/A', 'n/a', '']:
            return 0.0
        
        # Extract numeric value
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            try:
                price = float(match.group(1))
                # Sanity check
                if 0.01 <= price <= 100000:
                    return price
            except:
                pass
        
        return 0.0
    
    def scrape_managers_list(self):
        """Get list of all managers from Dataroma"""
        logging.info("Fetching managers list...")
        
        try:
            response = self.fetch_page(self.base_url + "managers.php")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Debug save
            self.debug_save_html(response.text, "managers_page.html")
            
            managers = []
            
            # Look for manager links - updated patterns
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Match holdings.php?m=XXX pattern
                if 'holdings.php?m=' in href:
                    # Extract manager ID
                    match = re.search(r'holdings\.php\?m=(\w+)', href)
                    if match:
                        manager_id = match.group(1)
                        manager_name = self.clean_text(link.text) or f"Manager_{manager_id}"
                        
                        # Build full URL
                        if href.startswith('http'):
                            full_url = href
                        elif href.startswith('/'):
                            full_url = f"https://www.dataroma.com{href}"
                        else:
                            full_url = f"{self.base_url}{href}"
                        
                        # Avoid duplicates
                        if not any(m['id'] == manager_id for m in managers):
                            managers.append({
                                'id': manager_id,
                                'name': manager_name,
                                'url': full_url
                            })
            
            logging.info(f"Found {len(managers)} managers")
            return managers
            
        except Exception as e:
            logging.error(f"Error fetching managers: {e}")
            return []
    
    def parse_holdings_table(self, soup, manager_name, portfolio_date):
        """Parse holdings table with better column detection"""
        holdings = []
        
        # Find the main holdings table
        holdings_table = None
        
        # Try different table selectors
        selectors = [
            {'id': 'grid'},
            {'class': 'grid'},
            {'id': 't1'},
            lambda tag: tag.name == 'table' and tag.find('a', href=lambda x: x and 'stock.php' in str(x))
        ]
        
        for selector in selectors:
            if isinstance(selector, dict):
                holdings_table = soup.find('table', selector)
            else:
                holdings_table = soup.find(selector)
            
            if holdings_table:
                break
        
        if not holdings_table:
            logging.warning(f"No holdings table found for {manager_name}")
            return holdings
        
        rows = holdings_table.find_all('tr')
        
        # Parse header row to map columns
        headers = []
        header_map = {}
        
        for row_idx, row in enumerate(rows):
            # Check for header row
            if row.find('th'):
                cells = row.find_all(['th', 'td'])
                for col_idx, cell in enumerate(cells):
                    header_text = self.clean_text(cell.text).lower()
                    headers.append(header_text)
                    
                    # Map columns based on keywords
                    if any(word in header_text for word in ['stock', 'company', 'holding', 'name']):
                        header_map['stock'] = col_idx
                    elif any(word in header_text for word in ['%', 'percent', 'portfolio']) and 'change' not in header_text:
                        header_map['percent'] = col_idx
                    elif 'shares' in header_text and 'change' not in header_text:
                        header_map['shares'] = col_idx
                    elif any(word in header_text for word in ['activity', 'recent', 'change']):
                        header_map['activity'] = col_idx
                    elif 'reported' in header_text and 'price' in header_text:
                        header_map['reported_price'] = col_idx
                    elif 'current' in header_text and 'price' in header_text:
                        header_map['current_price'] = col_idx
                    elif 'value' in header_text and '$' not in header_text:
                        header_map['value'] = col_idx
                
                logging.debug(f"Headers found: {headers}")
                logging.debug(f"Header mapping: {header_map}")
                break
        
        # Process data rows
        for row in rows:
            if row.find('th'):  # Skip header rows
                continue
            
            cells = row.find_all('td')
            if not cells:
                continue
            
            # Find stock link and ticker
            stock_link = None
            ticker = None
            company_name = ""
            
            # Look for stock link
            for cell in cells:
                link = cell.find('a', href=lambda x: x and 'stock.php' in str(x))
                if link:
                    stock_link = link
                    href = link.get('href', '')
                    ticker_match = re.search(r'sym=([A-Z0-9\.\-]+)', href, re.IGNORECASE)
                    if ticker_match:
                        ticker = ticker_match.group(1).upper()
                        company_name = self.clean_text(link.text)
                        break
            
            if not ticker:
                continue
            
            # Extract data based on column mapping
            holding = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'portfolio_date': portfolio_date,
                'manager': manager_name,
                'ticker': ticker,
                'company': company_name,
                'stock': company_name,  # For compatibility
                'portfolio_percent': 0.0,
                'shares': 0,
                'recent_activity': "",
                'reported_price': 0.0,
                'current_price': 0.0,
                'value': 0.0
            }
            
            # Extract percentage
            if 'percent' in header_map and header_map['percent'] < len(cells):
                percent_text = cells[header_map['percent']].text
                holding['portfolio_percent'] = self.extract_percentage(percent_text)
            
            # Extract shares
            if 'shares' in header_map and header_map['shares'] < len(cells):
                shares_text = cells[header_map['shares']].text
                holding['shares'] = self.extract_number(shares_text)
            
            # Extract activity
            if 'activity' in header_map and header_map['activity'] < len(cells):
                activity = self.clean_text(cells[header_map['activity']].text)
                if activity and activity not in ['-', 'N/A']:
                    holding['recent_activity'] = activity
            
            # Extract prices
            if 'reported_price' in header_map and header_map['reported_price'] < len(cells):
                holding['reported_price'] = self.extract_price(cells[header_map['reported_price']].text)
            
            if 'current_price' in header_map and header_map['current_price'] < len(cells):
                holding['current_price'] = self.extract_price(cells[header_map['current_price']].text)
            
            # If we still don't have portfolio %, try to find it in any cell
            if holding['portfolio_percent'] == 0.0:
                for cell in cells:
                    cell_text = cell.text
                    if '%' in cell_text:
                        pct = self.extract_percentage(cell_text)
                        if pct > 0:
                            holding['portfolio_percent'] = pct
                            break
            
            holdings.append(holding)
        
        return holdings
    
    def scrape_manager_holdings(self, manager_info):
        """Scrape holdings for a specific manager"""
        manager_url = manager_info['url']
        manager_name = manager_info['name']
        
        logging.info(f"Scraping holdings for {manager_name}")
        
        try:
            response = self.fetch_page(manager_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Debug save first manager
            if manager_info.get('id') in ['BL', 'WB', 'am']:  # Save a few for debugging
                self.debug_save_html(response.text, f"manager_{manager_info['id']}.html")
            
            # Extract portfolio date
            portfolio_date = self.extract_portfolio_date(soup)
            
            # Parse holdings
            holdings = self.parse_holdings_table(soup, manager_name, portfolio_date)
            
            logging.info(f"Found {len(holdings)} holdings for {manager_name}")
            
            # Log sample holding for debugging
            if holdings:
                sample = holdings[0]
                logging.debug(f"Sample holding: {sample['ticker']} - {sample['portfolio_percent']}%")
            
            return holdings
            
        except Exception as e:
            logging.error(f"Error scraping {manager_name}: {e}")
            return []
    
    def extract_portfolio_date(self, soup):
        """Extract portfolio date from page"""
        portfolio_date = datetime.now().strftime('%Y-%m-%d')
        
        # Look for date patterns in page text
        page_text = soup.get_text()
        
        patterns = [
            r'Portfolio\s*date:\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'Period:\s*Q\d\s+\d{4}.*?date:\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'Updated:\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'As\s+of\s+(\d{1,2}\s+\w+\s+\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    portfolio_date = datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
                    break
                except:
                    pass
        
        return portfolio_date
    
    def scrape_stock_data(self, ticker):
        """Enhanced stock data scraping"""
        # Check cache first
        if ticker in self.stock_cache:
            cached = self.stock_cache[ticker]
            if 'last_updated' in cached:
                last_updated = datetime.fromisoformat(cached['last_updated'])
                if datetime.now() - last_updated < timedelta(days=30):
                    return cached
        
        stock_url = f"{self.base_url}stock.php?sym={ticker}"
        logging.info(f"Fetching stock data for {ticker}")
        
        try:
            response = self.fetch_page(stock_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            stock_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat()
            }
            
            # Extract company name from h1
            h1 = soup.find('h1')
            if h1:
                company_text = self.clean_text(h1.text)
                # Remove ticker from name
                company_name = re.sub(r'\s*\([A-Z\.\-]+\)\s*$', '', company_text)
                if company_name:
                    stock_data['company_name'] = company_name
            
            # Parse all tables for data
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = self.clean_text(cells[0].text).lower()
                        value = self.clean_text(cells[1].text)
                        
                        # Extract different fields
                        if 'sector' in label:
                            stock_data['sector'] = value or 'Unknown'
                        elif 'industry' in label:
                            stock_data['industry'] = value or 'Unknown'
                        elif 'ownership count' in label:
                            stock_data['ownership_count'] = self.extract_number(value)
                        elif 'price' in label and 'current' in label:
                            stock_data['current_price'] = self.extract_price(value)
                        elif 'market cap' in label:
                            stock_data['market_cap'] = self.parse_market_cap(value)
                        elif 'p/e' in label or 'pe ratio' in label:
                            stock_data['pe_ratio'] = self.extract_price(value)
            
            # Update caches
            self.stock_cache[ticker] = stock_data
            
            # Update price cache
            if stock_data.get('current_price', 0) > 0:
                self.price_cache[ticker] = {
                    'price': stock_data['current_price'],
                    'updated': datetime.now().isoformat()
                }
            
            return stock_data
            
        except Exception as e:
            logging.error(f"Error fetching stock data for {ticker}: {e}")
            return {'ticker': ticker, 'sector': 'Unknown', 'industry': 'Unknown'}
    
    def parse_market_cap(self, text):
        """Parse market cap from text like $123.45B"""
        if not text:
            return 0
        
        text = text.replace('$', '').replace(',', '').strip()
        
        match = re.search(r'([\d.]+)\s*([BMKT])?', text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                multiplier = match.group(2)
                
                if multiplier:
                    multiplier = multiplier.upper()
                    if multiplier == 'B':
                        value *= 1_000_000_000
                    elif multiplier == 'M':
                        value *= 1_000_000
                    elif multiplier == 'K':
                        value *= 1_000
                    elif multiplier == 'T':
                        value *= 1_000_000_000_000
                
                return value
            except:
                pass
        
        return 0
    
    def enrich_holdings(self, holdings):
        """Enrich holdings with stock data"""
        logging.info(f"Enriching {len(holdings)} holdings with stock data...")
        
        # Get unique tickers
        unique_tickers = list(set(h['ticker'] for h in holdings if h.get('ticker')))
        
        # Fetch stock data for each ticker
        for i, ticker in enumerate(unique_tickers):
            if i % 10 == 0:
                logging.info(f"Progress: {i}/{len(unique_tickers)} stocks")
            
            stock_data = self.scrape_stock_data(ticker)
            
            # Update all holdings for this ticker
            for holding in holdings:
                if holding.get('ticker') == ticker:
                    # Add stock data to holding
                    holding.update({
                        'company_name': stock_data.get('company_name', holding.get('company', '')),
                        'sector': stock_data.get('sector', 'Unknown'),
                        'industry': stock_data.get('industry', 'Unknown'),
                        'market_cap': stock_data.get('market_cap', 0),
                        'pe_ratio': stock_data.get('pe_ratio', 0),
                        'ownership_count': stock_data.get('ownership_count', 0)
                    })
                    
                    # Update current price if we don't have it
                    if holding.get('current_price', 0) == 0 and stock_data.get('current_price', 0) > 0:
                        holding['current_price'] = stock_data['current_price']
            
            # Rate limiting
            time.sleep(0.5)
        
        return holdings
    
    def save_holdings(self, holdings):
        """Save holdings and generate summary"""
        if not holdings:
            logging.warning("No holdings to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(holdings)
        
        # Save files
        today = datetime.now().strftime('%Y%m%d')
        daily_file = os.path.join(self.holdings_dir, f'holdings_{today}.csv')
        
        df.to_csv(daily_file, index=False)
        df.to_csv('latest_holdings.csv', index=False)
        
        # Save caches
        self.save_cache(self.stock_cache, 'stock_data_cache.json')
        self.save_cache(self.price_cache, 'price_cache.json')
        
        # Generate metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_holdings': len(holdings),
            'unique_managers': len(df['manager'].unique()),
            'unique_stocks': len(df['ticker'].unique()),
            'holdings_with_percent': int((df['portfolio_percent'] > 0).sum()),
            'holdings_with_price': int((df['current_price'] > 0).sum()),
            'holdings_with_sector': int((df['sector'] != 'Unknown').sum()),
            'average_portfolio_percent': float(df[df['portfolio_percent'] > 0]['portfolio_percent'].mean()) if (df['portfolio_percent'] > 0).any() else 0,
            'data_quality': {
                'pct_with_portfolio_percent': round((df['portfolio_percent'] > 0).sum() / len(df) * 100, 1) if len(df) > 0 else 0,
                'pct_with_price': round((df['current_price'] > 0).sum() / len(df) * 100, 1) if len(df) > 0 else 0,
                'pct_with_sector': round((df['sector'] != 'Unknown').sum() / len(df) * 100, 1) if len(df) > 0 else 0
            }
        }
        
        with open('scrape_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"\nScraping Summary:")
        logging.info(f"Total holdings: {metadata['total_holdings']}")
        logging.info(f"Unique managers: {metadata['unique_managers']}")
        logging.info(f"Unique stocks: {metadata['unique_stocks']}")
        logging.info(f"Holdings with portfolio %: {metadata['holdings_with_percent']} ({metadata['data_quality']['pct_with_portfolio_percent']}%)")
        logging.info(f"Holdings with price: {metadata['holdings_with_price']} ({metadata['data_quality']['pct_with_price']}%)")
        logging.info(f"Average position size: {metadata['average_portfolio_percent']:.2f}%")
    
    def run(self):
        """Main scraping workflow"""
        logging.info("="*60)
        logging.info("IMPROVED DATAROMA SCRAPER")
        logging.info("="*60)
        
        # Get managers
        managers = self.scrape_managers_list()
        if not managers:
            logging.error("No managers found!")
            return
        
        # Scrape each manager
        all_holdings = []
        successful = 0
        
        for i, manager in enumerate(managers, 1):
            logging.info(f"\n[{i}/{len(managers)}] Processing {manager['name']}")
            
            holdings = self.scrape_manager_holdings(manager)
            if holdings:
                all_holdings.extend(holdings)
                successful += 1
                
                # Log data quality for this manager
                with_pct = sum(1 for h in holdings if h['portfolio_percent'] > 0)
                logging.info(f"  ✓ {len(holdings)} holdings ({with_pct} with %)")
            else:
                logging.warning(f"  ✗ No holdings found")
            
            # Rate limiting
            time.sleep(1)
            if i % 10 == 0:
                time.sleep(5)
        
        logging.info(f"\nManager scraping complete: {successful}/{len(managers)} successful")
        
        # Enrich with stock data
        if all_holdings:
            all_holdings = self.enrich_holdings(all_holdings)
        
        # Save results
        self.save_holdings(all_holdings)
        
        logging.info("\nScraping complete!")

def main():
    scraper = ImprovedDataromaScraper()
    try:
        scraper.run()
    except KeyboardInterrupt:
        logging.info("\nScraping interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())