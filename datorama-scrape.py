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

class DataromaScraperImproved:
    def __init__(self, debug_mode=True):
        self.base_url = "https://www.dataroma.com/m/"
        self.holdings_dir = "holdings"
        self.cache_dir = "cache"
        self.debug_dir = "debug"
        self.debug_mode = debug_mode
        self.ensure_directories()
        self.stock_cache = self.load_stock_cache()
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if debug_mode else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dataroma_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Enhanced column mappings with multiple possible headers
        self.column_mappings = {
            'stock': ['stock', 'company', 'holding', 'security', 'name', 'investment'],
            'percent': ['% of portfolio', 'portfolio %', 'weight', 'position', '%', 'percentage', 
                       'portfolio', 'allocation', '% weight', 'position size'],
            'shares': ['shares', 'quantity', 'amount', 'holdings', 'position', 'units'],
            'activity': ['recent activity', 'activity', 'action', 'change', 'recent', 'transaction'],
            'reported_price': ['reported price', 'cost', 'purchase price', 'avg cost', 'price paid'],
            'current_price': ['current price', 'price', 'last', 'current', 'market price'],
            'value': ['value', 'market value', 'current value', 'position value', 'total']
        }
        
        # Track extraction statistics
        self.extraction_stats = {
            'managers_processed': 0,
            'holdings_found': 0,
            'holdings_with_percent': 0,
            'holdings_with_shares': 0,
            'holdings_with_price': 0,
            'stocks_enriched': 0
        }
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.holdings_dir, self.cache_dir, self.debug_dir, "analysis"]:
            os.makedirs(directory, exist_ok=True)
    
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
        if self.debug_mode:
            debug_file = os.path.join(self.debug_dir, filename)
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            self.logger.debug(f"Saved HTML to {debug_file}")
    
    def debug_save_json(self, data, filename):
        """Save JSON data for debugging"""
        if self.debug_mode:
            debug_file = os.path.join(self.debug_dir, filename)
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Saved JSON to {debug_file}")
    
    def detect_table_structure(self, table):
        """Analyze and detect table structure"""
        structure = {
            'headers': [],
            'column_map': {},
            'sample_rows': []
        }
        
        rows = table.find_all('tr')
        
        # Find header row
        header_row_idx = -1
        for idx, row in enumerate(rows):
            ths = row.find_all('th')
            if ths:
                header_row_idx = idx
                # Extract all header text
                for i, th in enumerate(ths):
                    header_text = ' '.join(th.stripped_strings)
                    structure['headers'].append({
                        'index': i,
                        'text': header_text,
                        'text_lower': header_text.lower()
                    })
                break
        
        # Map columns based on headers
        for header_info in structure['headers']:
            i = header_info['index']
            header_lower = header_info['text_lower']
            
            # Try to match with known column types
            for col_type, keywords in self.column_mappings.items():
                if any(keyword in header_lower for keyword in keywords):
                    structure['column_map'][col_type] = i
                    self.logger.debug(f"Mapped column {i} '{header_info['text']}' to {col_type}")
                    break
        
        # Get sample data rows
        data_rows = rows[header_row_idx + 1:header_row_idx + 4] if header_row_idx >= 0 else rows[:3]
        for row in data_rows:
            cells = row.find_all('td')
            if cells:
                row_data = []
                for cell in cells:
                    # Get text and any links
                    cell_text = cell.get_text(strip=True)
                    link = cell.find('a')
                    row_data.append({
                        'text': cell_text,
                        'has_link': link is not None,
                        'link_href': link.get('href', '') if link else ''
                    })
                structure['sample_rows'].append(row_data)
        
        return structure
    
    def extract_percentage(self, text):
        """Enhanced percentage extraction"""
        if not text:
            return None
        
        # Clean the text
        text = str(text).strip().replace('*', '').replace(',', '.')
        
        # Try multiple patterns
        patterns = [
            r'(\d+\.?\d*)\s*%',       # 12.5% or 12%
            r'^(\d+\.?\d*)$',          # Just a number
            r'(\d+\.?\d*)\s*percent',  # 12.5 percent
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 100:  # Sanity check
                        return val
                except:
                    pass
        
        # Try to parse as direct number if it's small (likely a percentage)
        try:
            val = float(text)
            if 0 <= val <= 100:
                return val
        except:
            pass
        
        return None
    
    def extract_number(self, text):
        """Enhanced number extraction for shares"""
        if not text:
            return "0"
        
        # Remove common formatting
        text = str(text).replace(',', '').replace(' ', '').replace('$', '').strip()
        
        # Check for multipliers (K, M, B)
        multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
        for suffix, multiplier in multipliers.items():
            if text.lower().endswith(suffix):
                try:
                    return str(int(float(text[:-1]) * multiplier))
                except:
                    pass
        
        # Try to extract plain number
        match = re.search(r'(\d+)', text)
        if match:
            return match.group(1)
        
        return "0"
    
    def clean_price(self, price_str):
        """Enhanced price cleaning"""
        if not price_str:
            return None
        
        try:
            # Remove common non-numeric characters
            cleaned = str(price_str).replace('$', '').replace(',', '').replace(' ', '').strip()
            
            # Handle special cases
            if cleaned.lower() in ['-', 'n/a', 'na', 'none', '']:
                return None
            
            # Extract numeric value with decimal
            match = re.search(r'(\d+\.?\d*)', cleaned)
            if match:
                price = float(match.group(1))
                # Sanity check for reasonable stock prices
                if 0.01 <= price <= 100000:
                    return price
        except Exception as e:
            self.logger.debug(f"Price extraction failed for '{price_str}': {e}")
        
        return None
    
    def scrape_manager_enhanced(self, manager_link, hint_name=""):
        """Enhanced manager scraping with better debugging"""
        # Handle URLs
        if manager_link.startswith('/'):
            manager_url = "https://www.dataroma.com" + manager_link
        elif not manager_link.startswith('http'):
            manager_url = self.base_url + manager_link.lstrip('/')
        else:
            manager_url = manager_link
        
        self.logger.info(f"Scraping manager: {manager_url}")
        
        try:
            response = requests.get(manager_url, timeout=30, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {manager_url}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Save HTML for debugging
        safe_name = re.sub(r'[^\w\-_]', '_', hint_name or 'unknown')
        self.debug_save_html(response.text, f"manager_{safe_name}.html")
        
        holdings = []
        
        # Extract manager name
        manager_name = hint_name
        h1_elem = soup.find('h1')
        if h1_elem and h1_elem.text.strip():
            manager_name = h1_elem.text.strip()
            self.logger.debug(f"Manager name: {manager_name}")
        
        # Extract portfolio date
        portfolio_date = datetime.now().strftime('%Y-%m-%d')
        date_patterns = [
            r'Portfolio\s*date:\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'As\s*of\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'Date:\s*(\d{1,2}\s+\w+\s+\d{4})',
            r'Updated:\s*(\d{1,2}\s+\w+\s+\d{4})'
        ]
        
        page_text = soup.get_text()
        for pattern in date_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    portfolio_date = datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
                    self.logger.debug(f"Portfolio date: {portfolio_date}")
                    break
                except:
                    pass
        
        # Find holdings table - try multiple strategies
        holdings_table = None
        
        # Strategy 1: Look for table with id='grid'
        holdings_table = soup.find('table', {'id': 'grid'})
        
        # Strategy 2: Look for table containing stock.php links
        if not holdings_table:
            tables = soup.find_all('table')
            for table in tables:
                if table.find('a', href=lambda x: x and 'stock.php' in x):
                    holdings_table = table
                    self.logger.debug("Found holdings table by stock.php links")
                    break
        
        # Strategy 3: Look for largest table
        if not holdings_table and tables:
            holdings_table = max(tables, key=lambda t: len(t.find_all('tr')))
            self.logger.debug("Using largest table as holdings table")
        
        if holdings_table:
            # Analyze table structure
            structure = self.detect_table_structure(holdings_table)
            self.debug_save_json(structure, f"table_structure_{safe_name}.json")
            
            # If debugging, print the structure
            if self.debug_mode:
                self.logger.debug(f"Table headers: {[h['text'] for h in structure['headers']]}")
                self.logger.debug(f"Column mapping: {structure['column_map']}")
                
                # Print first few rows
                for i, row in enumerate(structure['sample_rows'][:3]):
                    self.logger.debug(f"Sample row {i}: {[cell['text'] for cell in row]}")
            
            # Process data rows
            rows = holdings_table.find_all('tr')
            header_row_idx = -1
            
            # Find header row index
            for idx, row in enumerate(rows):
                if row.find_all('th'):
                    header_row_idx = idx
                    break
            
            # Process each data row
            for row_idx, row in enumerate(rows):
                if row_idx <= header_row_idx:
                    continue
                
                cells = row.find_all('td')
                if not cells or len(cells) < 2:
                    continue
                
                holding = self.extract_holding_from_row(cells, structure['column_map'], 
                                                       manager_name, portfolio_date)
                
                if holding and holding.get('ticker'):
                    holdings.append(holding)
                    
                    # Update stats
                    if float(holding.get('portfolio_percent', 0)) > 0:
                        self.extraction_stats['holdings_with_percent'] += 1
                    if int(holding.get('shares', '0')) > 0:
                        self.extraction_stats['holdings_with_shares'] += 1
        else:
            self.logger.warning(f"No holdings table found for {manager_name}")
        
        self.logger.info(f"Found {len(holdings)} holdings for {manager_name}")
        return holdings
    
    def extract_holding_from_row(self, cells, col_map, manager_name, portfolio_date):
        """Extract holding data from a table row with fallback strategies"""
        holding = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_date': portfolio_date,
            'manager': manager_name,
            'stock': '',
            'ticker': '',
            'portfolio_percent': "0",
            'shares': "0",
            'recent_activity': "",
            'reported_price': 0,
            'current_price': 0,
            'value': 0
        }
        
        # First, try to find stock ticker and name
        stock_link = None
        ticker = None
        
        # Try mapped column first
        if 'stock' in col_map and col_map['stock'] < len(cells):
            cell = cells[col_map['stock']]
            stock_link = cell.find('a', href=lambda x: x and 'stock.php' in x)
        
        # Search all cells for stock link if not found
        if not stock_link:
            for i, cell in enumerate(cells):
                link = cell.find('a', href=lambda x: x and 'stock.php' in x)
                if link:
                    stock_link = link
                    # Update column mapping for future rows
                    col_map['stock'] = i
                    break
        
        if stock_link:
            # Extract ticker from URL
            href = stock_link.get('href', '')
            ticker_match = re.search(r'[?&]sym=([A-Z0-9.-]+)', href, re.IGNORECASE)
            if ticker_match:
                ticker = ticker_match.group(1).upper()
                holding['ticker'] = ticker
                
                # Extract stock name
                stock_name = stock_link.text.strip()
                # Remove ticker from name if present
                stock_name = re.sub(r'^[A-Z0-9.-]+\s*[-–]\s*', '', stock_name).strip()
                holding['stock'] = stock_name
        
        if not ticker:
            return None
        
        # Extract percentage - try multiple strategies
        percent_found = False
        
        # Strategy 1: Use mapped column
        if 'percent' in col_map and col_map['percent'] < len(cells):
            percent_text = cells[col_map['percent']].text.strip()
            percent_val = self.extract_percentage(percent_text)
            if percent_val is not None:
                holding['portfolio_percent'] = str(percent_val)
                percent_found = True
        
        # Strategy 2: Look for percentage in any cell
        if not percent_found:
            for i, cell in enumerate(cells):
                cell_text = cell.text.strip()
                # Look for percentage pattern
                if re.search(r'\d+\.?\d*\s*%', cell_text):
                    percent_val = self.extract_percentage(cell_text)
                    if percent_val is not None and 0 < percent_val <= 100:
                        holding['portfolio_percent'] = str(percent_val)
                        col_map['percent'] = i  # Update mapping
                        percent_found = True
                        self.logger.debug(f"Found percentage {percent_val}% in column {i}")
                        break
        
        # Extract shares
        if 'shares' in col_map and col_map['shares'] < len(cells):
            shares_text = cells[col_map['shares']].text.strip()
            holding['shares'] = self.extract_number(shares_text)
        
        # Extract activity
        if 'activity' in col_map and col_map['activity'] < len(cells):
            activity_text = cells[col_map['activity']].text.strip()
            if activity_text and activity_text != '-':
                holding['recent_activity'] = activity_text
        
        # Extract prices
        if 'reported_price' in col_map and col_map['reported_price'] < len(cells):
            price_text = cells[col_map['reported_price']].text.strip()
            price = self.clean_price(price_text)
            if price:
                holding['reported_price'] = price
        
        if 'current_price' in col_map and col_map['current_price'] < len(cells):
            price_text = cells[col_map['current_price']].text.strip()
            price = self.clean_price(price_text)
            if price:
                holding['current_price'] = price
                self.extraction_stats['holdings_with_price'] += 1
        
        # Extract value
        if 'value' in col_map and col_map['value'] < len(cells):
            value_text = cells[col_map['value']].text.strip()
            value = self.extract_value(value_text)
            if value:
                holding['value'] = value
        
        return holding
    
    def extract_value(self, text):
        """Extract value which might be in thousands or millions"""
        if not text:
            return 0
        
        # Remove $ and commas
        text = text.replace('$', '').replace(',', '').strip()
        
        try:
            # Check for multipliers
            multipliers = {
                't': 1_000_000_000_000,  # Trillion
                'b': 1_000_000_000,      # Billion
                'm': 1_000_000,          # Million
                'k': 1_000               # Thousand
            }
            
            for suffix, multiplier in multipliers.items():
                if text.lower().endswith(suffix):
                    return float(text[:-1]) * multiplier
            
            # Try as plain number
            return float(text)
        except:
            return 0
    
    def scrape_stock_page_enhanced(self, ticker):
        """Enhanced stock page scraping with better pattern matching"""
        # Check cache first
        if ticker in self.stock_cache and not self.should_refresh_stock(ticker):
            return self.stock_cache[ticker]
        
        stock_url = f"{self.base_url}stock.php?sym={ticker}"
        self.logger.debug(f"Fetching stock data for {ticker}")
        
        try:
            response = requests.get(stock_url, timeout=10, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Save HTML for debugging
            if self.debug_mode:
                self.debug_save_html(response.text, f"stock_{ticker}.html")
            
            stock_data = {
                'ticker': ticker,
                'last_updated': datetime.now().isoformat()
            }
            
            # Extract company name
            h1 = soup.find('h1')
            if h1:
                company_text = h1.text.strip()
                company_name = re.sub(r'\s*\([A-Z]+\)\s*$', '', company_text).strip()
                if company_name:
                    stock_data['company_name'] = company_name
            
            # Extract all table data
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].text.strip()
                        value = cells[1].text.strip()
                        
                        label_lower = label.lower()
                        
                        # Map various labels to data fields
                        if 'sector' in label_lower:
                            stock_data['sector'] = value if value else 'Unknown'
                        elif 'industry' in label_lower:
                            stock_data['industry'] = value if value else 'Unknown'
                        elif 'ownership count' in label_lower:
                            try:
                                stock_data['ownership_count'] = int(re.search(r'\d+', value).group())
                            except:
                                pass
                        elif 'ownership rank' in label_lower:
                            try:
                                stock_data['ownership_rank'] = int(re.search(r'\d+', value).group())
                            except:
                                pass
                        elif '% of all portfolios' in label_lower:
                            pct = self.extract_percentage(value)
                            if pct is not None:
                                stock_data['pct_of_all_portfolios'] = pct
                        elif 'hold price' in label_lower or 'avg price' in label_lower:
                            price = self.clean_price(value)
                            if price:
                                stock_data['hold_price'] = price
                        elif 'current price' in label_lower or label_lower == 'price':
                            price = self.clean_price(value)
                            if price:
                                stock_data['current_price'] = price
            
            # Try to extract additional data from page text
            page_text = soup.get_text()
            
            # Market cap patterns
            market_cap_patterns = [
                r'Market\s*Cap[:\s]+\$?([\d,]+\.?\d*)\s*([BMKT])',
                r'Mkt\s*Cap[:\s]+\$?([\d,]+\.?\d*)\s*([BMKT])',
                r'Capitalization[:\s]+\$?([\d,]+\.?\d*)\s*([BMKT])'
            ]
            
            for pattern in market_cap_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1).replace(',', ''))
                    multiplier_char = match.group(2).upper()
                    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
                    if multiplier_char in multipliers:
                        stock_data['market_cap'] = value * multipliers[multiplier_char]
                        break
            
            # Fill defaults
            if 'sector' not in stock_data:
                stock_data['sector'] = 'Unknown'
            if 'industry' not in stock_data:
                stock_data['industry'] = 'Unknown'
            
            # Use hold price as current price if we don't have current price
            if 'current_price' not in stock_data and 'hold_price' in stock_data:
                stock_data['current_price'] = stock_data['hold_price']
            
            # Update cache
            self.stock_cache[ticker] = stock_data
            self.extraction_stats['stocks_enriched'] += 1
            
            # Small delay
            time.sleep(0.5)
            
            return stock_data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock page for {ticker}: {e}")
            # Return cached data if available
            return self.stock_cache.get(ticker, {
                'ticker': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown'
            })
    
    def should_refresh_stock(self, ticker):
        """Check if stock data should be refreshed"""
        if ticker not in self.stock_cache:
            return True
        
        cached_data = self.stock_cache[ticker]
        if 'last_updated' in cached_data:
            last_updated = datetime.fromisoformat(cached_data['last_updated'])
            # Update monthly
            if datetime.now() - last_updated > timedelta(days=30):
                return True
        
        return False
    
    def validate_and_report_data(self, all_holdings):
        """Validate data quality and print detailed report"""
        self.logger.info("\n" + "="*60)
        self.logger.info("DATA EXTRACTION VALIDATION REPORT")
        self.logger.info("="*60)
        
        total = len(all_holdings)
        if total == 0:
            self.logger.warning("No holdings data extracted!")
            return False
        
        # Count various data points
        with_percent = sum(1 for h in all_holdings if float(h.get('portfolio_percent', 0)) > 0)
        with_shares = sum(1 for h in all_holdings if int(h.get('shares', '0')) > 0)
        with_price = sum(1 for h in all_holdings if float(h.get('current_price', 0)) > 0)
        with_reported_price = sum(1 for h in all_holdings if float(h.get('reported_price', 0)) > 0)
        with_activity = sum(1 for h in all_holdings if h.get('recent_activity', '').strip())
        with_sector = sum(1 for h in all_holdings if h.get('sector', 'Unknown') != 'Unknown')
        with_market_cap = sum(1 for h in all_holdings if float(h.get('market_cap', 0)) > 0)
        
        # Print detailed statistics
        self.logger.info(f"Total holdings extracted: {total}")
        self.logger.info(f"Holdings with portfolio %: {with_percent} ({with_percent/total*100:.1f}%)")
        self.logger.info(f"Holdings with shares: {with_shares} ({with_shares/total*100:.1f}%)")
        self.logger.info(f"Holdings with current price: {with_price} ({with_price/total*100:.1f}%)")
        self.logger.info(f"Holdings with reported price: {with_reported_price} ({with_reported_price/total*100:.1f}%)")
        self.logger.info(f"Holdings with recent activity: {with_activity} ({with_activity/total*100:.1f}%)")
        self.logger.info(f"Holdings with sector: {with_sector} ({with_sector/total*100:.1f}%)")
        self.logger.info(f"Holdings with market cap: {with_market_cap} ({with_market_cap/total*100:.1f}%)")
        
        # Sample of data for debugging
        if self.debug_mode and all_holdings:
            self.logger.info("\nSample holdings (first 5 with portfolio %):")
            sample = [h for h in all_holdings if float(h.get('portfolio_percent', 0)) > 0][:5]
            for h in sample:
                self.logger.info(f"  {h['manager']} - {h['ticker']}: {h['portfolio_percent']}%")
        
        # Warnings for missing data
        if with_percent == 0:
            self.logger.warning("\n⚠️  WARNING: No portfolio percentage data found!")
            self.logger.warning("This suggests the column mapping for percentages is failing.")
            self.logger.warning("Check the debug/table_structure_*.json files to see actual headers.")
        
        if with_price == 0:
            self.logger.warning("\n⚠️  WARNING: No price data found!")
        
        # Return success if we have meaningful data
        return with_percent > 0 or with_shares > 0
    
    def scrape_dataroma(self):
        """Main scraping function with enhanced debugging"""
        self.logger.info("="*60)
        self.logger.info("DATAROMA SCRAPER - ENHANCED VERSION")
        self.logger.info("="*60)
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Debug mode: {self.debug_mode}")
        
        # Reset stats
        self.extraction_stats = {
            'managers_processed': 0,
            'holdings_found': 0,
            'holdings_with_percent': 0,
            'holdings_with_shares': 0,
            'holdings_with_price': 0,
            'stocks_enriched': 0
        }
        
        # Get list of managers
        managers_url = self.base_url + "managers.php"
        self.logger.info(f"\nFetching managers from: {managers_url}")
        
        try:
            response = requests.get(managers_url, timeout=30, headers=self.headers)
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching managers page: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Save managers page for debugging
        self.debug_save_html(response.text, "managers_page.html")
        
        all_holdings = []
        manager_links = []
        
        # Extract manager links - ONLY get holdings.php?m= links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'holdings.php?m=' in href and 'stock.php' not in href:
                manager_info = {
                    'url': href,
                    'name': link.text.strip() if link.text else ""
                }
                if not any(m['url'] == manager_info['url'] for m in manager_links):
                    manager_links.append(manager_info)
        
        self.logger.info(f"Found {len(manager_links)} managers to scrape")
        
        # Process managers (limit to first 5 in debug mode for faster testing)
        if self.debug_mode:
            manager_links = manager_links[:5]
            self.logger.info(f"Debug mode: Processing only first 5 managers")
        
        successful_managers = 0
        failed_managers = 0
        
        for idx, manager_info in enumerate(manager_links, 1):
            try:
                self.logger.info(f"\n[{idx}/{len(manager_links)}] Processing: {manager_info['name']}")
                holdings = self.scrape_manager_enhanced(manager_info['url'], manager_info.get('name', ''))
                
                if holdings:
                    all_holdings.extend(holdings)
                    successful_managers += 1
                    self.extraction_stats['managers_processed'] += 1
                    self.extraction_stats['holdings_found'] += len(holdings)
                    self.logger.info(f"  ✓ Found {len(holdings)} holdings")
                else:
                    failed_managers += 1
                    self.logger.warning(f"  ✗ No holdings found")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                failed_managers += 1
                self.logger.error(f"  ✗ Error: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
        
        # Validate data before enrichment
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PRE-ENRICHMENT VALIDATION")
        self.validate_and_report_data(all_holdings)
        
        # Get unique tickers for stock data enrichment
        unique_tickers = list(set(h['ticker'] for h in all_holdings if h.get('ticker')))
        self.logger.info(f"\nUnique stocks found: {len(unique_tickers)}")
        
        # Enrich with stock data (limit in debug mode)
        if unique_tickers:
            if self.debug_mode:
                unique_tickers = unique_tickers[:20]
                self.logger.info(f"Debug mode: Enriching only first 20 stocks")
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info("ENRICHING HOLDINGS WITH STOCK DATA")
            
            for i, ticker in enumerate(unique_tickers):
                if i % 10 == 0:
                    self.logger.info(f"Progress: {i}/{len(unique_tickers)} stocks processed")
                
                stock_data = self.scrape_stock_page_enhanced(ticker)
                
                # Update all holdings for this ticker
                for holding in all_holdings:
                    if holding.get('ticker') == ticker:
                        for key, value in stock_data.items():
                            if key not in ['ticker', 'last_updated']:
                                holding[key] = value
        
        # Save cache
        self.save_stock_cache()
        self.logger.info(f"\nStock cache updated with {len(self.stock_cache)} stocks")
        
        # Final validation
        self.logger.info(f"\n{'='*60}")
        self.logger.info("POST-ENRICHMENT VALIDATION")
        data_valid = self.validate_and_report_data(all_holdings)
        
        # Save holdings
        self.save_holdings(all_holdings)
        
        # Print final statistics
        self.logger.info(f"\n{'='*60}")
        self.logger.info("SCRAPING COMPLETE - FINAL STATISTICS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Managers processed: {self.extraction_stats['managers_processed']}")
        self.logger.info(f"Total holdings found: {self.extraction_stats['holdings_found']}")
        self.logger.info(f"Holdings with portfolio %: {self.extraction_stats['holdings_with_percent']}")
        self.logger.info(f"Holdings with shares: {self.extraction_stats['holdings_with_shares']}")
        self.logger.info(f"Holdings with price: {self.extraction_stats['holdings_with_price']}")
        self.logger.info(f"Stocks enriched: {self.extraction_stats['stocks_enriched']}")
        self.logger.info(f"Data validation: {'PASSED' if data_valid else 'FAILED'}")
        self.logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return all_holdings
    
    def save_holdings(self, all_holdings):
        """Save holdings with detailed metadata"""
        today = datetime.now().strftime('%Y%m%d')
        
        if not all_holdings:
            self.logger.warning("No holdings to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_holdings)
        
        # Save files
        daily_filename = os.path.join(self.holdings_dir, f'holdings_{today}.csv')
        df.to_csv(daily_filename, index=False)
        df.to_csv('latest_holdings.csv', index=False)
        
        # Create detailed metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'scrape_date': today,
            'extraction_stats': self.extraction_stats,
            'data_quality': {
                'total_holdings': len(all_holdings),
                'unique_managers': len(df['manager'].unique()) if 'manager' in df.columns else 0,
                'unique_stocks': len(df['ticker'].unique()) if 'ticker' in df.columns else 0,
                'pct_with_portfolio_percent': round(self.extraction_stats['holdings_with_percent'] / len(all_holdings) * 100, 1) if all_holdings else 0,
                'pct_with_shares': round(self.extraction_stats['holdings_with_shares'] / len(all_holdings) * 100, 1) if all_holdings else 0,
                'pct_with_price': round(self.extraction_stats['holdings_with_price'] / len(all_holdings) * 100, 1) if all_holdings else 0
            },
            'debug_info': {
                'debug_mode': self.debug_mode,
                'debug_files_created': os.listdir(self.debug_dir) if self.debug_mode else []
            }
        }
        
        with open('scrape_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"\nFiles saved:")
        self.logger.info(f"- {daily_filename}")
        self.logger.info(f"- latest_holdings.csv")
        self.logger.info(f"- scrape_metadata.json")


def main():
    """Main function with debug mode option"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Dataroma Scraper')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--limit-managers', type=int, help='Limit number of managers to scrape')
    args = parser.parse_args()
    
    # Create scraper with debug mode
    scraper = DataromaScraperImproved(debug_mode=args.debug)
    
    if args.limit_managers:
        scraper.logger.info(f"Limiting to {args.limit_managers} managers")
    
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
        all_holdings = scraper.scrape_dataroma()
        
        if args.debug and all_holdings:
            print(f"\n📁 Debug files saved to: {scraper.debug_dir}/")
            print("Check these files to understand the HTML structure:")
            print("- managers_page.html")
            print("- manager_*.html (individual manager pages)")
            print("- table_structure_*.json (extracted table structures)")
            print("- stock_*.html (stock pages)")
        
        return 0
    except Exception as e:
        scraper.logger.error(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())