import pandas as pd
import git
import os
import yfinance as yf
from datetime import datetime, timedelta
import json
import argparse
import glob
import numpy as np
from collections import defaultdict
import shutil
import time
import warnings
import logging

# Suppress all warnings and yfinance errors
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.ERROR)
# Suppress urllib3 errors
logging.getLogger('urllib3').setLevel(logging.ERROR)
# Redirect yfinance output
import sys
import io

class SmartHoldingsAnalyzer:
    def __init__(self):
        self.repo = git.Repo('.')
        self.price_cache = {}
        self.market_cap_cache = {}
        self.holdings_dir = "holdings"
        self.analysis_dir = "analysis"
        self.yfinance_enabled = False  # Disabled by default due to reliability issues
        self.yfinance_failures = 0
        self.max_yfinance_failures = 10
        self.ensure_directories()
        self.load_caches()
    
    def ensure_directories(self):
        """Ensure analysis directories exist"""
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(os.path.join(self.analysis_dir, 'master_stock_analysis'), exist_ok=True)
    
    def load_caches(self):
        """Load cached prices and market caps to avoid repeated API calls"""
        # Price cache
        cache_file = os.path.join(self.analysis_dir, 'price_cache.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.price_cache = json.load(f)
        
        # Market cap cache
        mcap_file = os.path.join(self.analysis_dir, 'market_cap_cache.json')
        if os.path.exists(mcap_file):
            with open(mcap_file, 'r') as f:
                self.market_cap_cache = json.load(f)
    
    def save_caches(self):
        """Save all caches for future use"""
        # Price cache
        cache_file = os.path.join(self.analysis_dir, 'price_cache.json')
        with open(cache_file, 'w') as f:
            json.dump(self.price_cache, f)
        
        # Market cap cache
        mcap_file = os.path.join(self.analysis_dir, 'market_cap_cache.json')
        with open(mcap_file, 'w') as f:
            json.dump(self.market_cap_cache, f)
    
    def get_stock_info(self, ticker):
        """Get comprehensive stock information including market cap"""
        cache_key = f"{ticker}_info"
        
        # Check if we have recent info (within 7 days)
        if cache_key in self.market_cap_cache:
            cached_info = self.market_cap_cache[cache_key]
            if 'timestamp' in cached_info:
                cache_time = datetime.fromisoformat(cached_info['timestamp'])
                if datetime.now() - cache_time < timedelta(days=7):
                    return cached_info
        
        # If yfinance is disabled, return default values
        if not self.yfinance_enabled:
            return {
                'market_cap': 0,
                'current_price': 0,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'company_name': ticker
            }
        
        # Suppress all output during yfinance calls
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            if info and isinstance(info, dict) and len(info) > 5:
                stock_data = {
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'company_name': info.get('longName', info.get('shortName', ticker)),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Only cache if we got meaningful data
                if stock_data['current_price'] > 0 or stock_data['market_cap'] > 0:
                    self.market_cap_cache[cache_key] = stock_data
                    return stock_data
                else:
                    self.yfinance_failures += 1
                
        except Exception as e:
            # Handle any errors (401, 404, network issues, etc.)
            self.yfinance_failures += 1
            
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Disable yfinance if too many failures
            if self.yfinance_failures >= self.max_yfinance_failures:
                print(f"\nWarning: Disabling Yahoo Finance data due to repeated errors. Continuing with limited functionality.")
                self.yfinance_enabled = False
        
        # Return default values if API fails
        return {
            'market_cap': 0,
            'current_price': 0,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'company_name': ticker
        }
    
    def categorize_market_cap(self, market_cap):
        """Categorize stocks by market cap"""
        if market_cap <= 0:
            return 'Unknown'
        elif market_cap < 300_000_000:
            return 'Micro-Cap'
        elif market_cap < 2_000_000_000:
            return 'Small-Cap'
        elif market_cap < 10_000_000_000:
            return 'Mid-Cap'
        elif market_cap < 200_000_000_000:
            return 'Large-Cap'
        else:
            return 'Mega-Cap'
    
    def get_historical_price(self, ticker, date):
        """Get historical price with caching"""
        cache_key = f"{ticker}_{date}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        # If yfinance is disabled, return None
        if not self.yfinance_enabled:
            return None
        
        # Suppress all output during yfinance calls
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=7)
            
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self.price_cache[cache_key] = price
                return price
        except Exception as e:
            # Handle errors gracefully
            self.yfinance_failures += 1
            
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Disable yfinance if too many failures
            if self.yfinance_failures >= self.max_yfinance_failures:
                print(f"\nWarning: Disabling Yahoo Finance data due to repeated errors. Continuing with limited functionality.")
                self.yfinance_enabled = False
        
        self.price_cache[cache_key] = None
        return None
    
    def get_latest_files(self, n=2):
        """Get the n most recent holdings files"""
        files = glob.glob(os.path.join(self.holdings_dir, 'holdings_*.csv'))
        files.sort(reverse=True)
        return files[:n]
    
    def analyze_current_holdings(self):
        """Analyze current holdings comprehensively"""
        print("Analyzing current holdings...")
        
        # Get latest holdings file
        latest_files = self.get_latest_files(1)
        if not latest_files:
            print("No holdings files found")
            return {}
        
        holdings_df = pd.read_csv(latest_files[0])
        
        # Debug: Check what columns we have
        print(f"Available columns: {list(holdings_df.columns)}")
        
        # Convert numeric columns with better error handling
        if 'shares' in holdings_df.columns:
            # Handle shares that might have commas
            holdings_df['shares_num'] = pd.to_numeric(
                holdings_df['shares'].astype(str).str.replace(',', '').str.strip(), 
                errors='coerce'
            ).fillna(0)
        else:
            holdings_df['shares_num'] = 0
        
        # Handle portfolio_percent column - it might be a string with % sign or already numeric
        if 'portfolio_percent' in holdings_df.columns:
            # First check if it's already numeric
            if pd.api.types.is_numeric_dtype(holdings_df['portfolio_percent']):
                holdings_df['percent_num'] = holdings_df['portfolio_percent'].fillna(0)
            else:
                # It's a string, need to clean it
                holdings_df['percent_num'] = pd.to_numeric(
                    holdings_df['portfolio_percent'].astype(str).str.replace('%', '').str.strip(), 
                    errors='coerce'
                ).fillna(0)
        else:
            holdings_df['percent_num'] = 0
        
        # Debug: Check conversion results
        print(f"\nData conversion summary:")
        print(f"- Portfolio % range: {holdings_df['percent_num'].min():.2f} to {holdings_df['percent_num'].max():.2f}")
        print(f"- Holdings with portfolio % > 0: {(holdings_df['percent_num'] > 0).sum()}")
        print(f"- Holdings with shares > 0: {(holdings_df['shares_num'] > 0).sum()}")
        
        # If we have very few valid portfolio percentages, show a warning
        if (holdings_df['percent_num'] > 0).sum() < 100:
            print("\nWarning: Very few holdings have valid portfolio percentages.")
            print("This may indicate an issue with the data scraping.")
            print("The analysis will proceed but conviction scores may be based primarily on manager count.")
        
        # Try to use prices from Dataroma if available
        holdings_df['dataroma_price'] = 0
        if 'reported_price' in holdings_df.columns:
            holdings_df['reported_price_num'] = pd.to_numeric(holdings_df['reported_price'], errors='coerce').fillna(0)
            holdings_df['dataroma_price'] = holdings_df['reported_price_num']
        if 'current_price' in holdings_df.columns:
            current_price_num = pd.to_numeric(holdings_df['current_price'], errors='coerce').fillna(0)
            holdings_df['dataroma_price'] = holdings_df['dataroma_price'].combine(current_price_num, max)
        
        # Get stock info for all unique tickers
        print("Processing stock information...")
        ticker_info = {}
        unique_tickers = holdings_df['ticker'].unique()
        
        # Only fetch external data if explicitly enabled
        if self.yfinance_enabled:
            print("Fetching external stock data...")
            for i, ticker in enumerate(unique_tickers):
                if i % 50 == 0:
                    print(f"  Processing {i}/{len(unique_tickers)} tickers...")
                
                ticker_info[ticker] = self.get_stock_info(ticker)
                # Small delay to avoid rate limiting
                if i % 10 == 0:
                    time.sleep(0.1)
        else:
            print("Using Dataroma data only (external data fetching disabled)")
            # Use Dataroma prices if available
            for ticker in unique_tickers:
                ticker_prices = holdings_df[holdings_df['ticker'] == ticker]
                price = ticker_prices['dataroma_price'].max() if 'dataroma_price' in holdings_df.columns else 0
                
                ticker_info[ticker] = {
                    'market_cap': 0,
                    'current_price': price if pd.notna(price) and price > 0 else 0,
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'company_name': ticker_prices['stock'].iloc[0] if 'stock' in holdings_df.columns and len(ticker_prices) > 0 else ticker
                }
        
        # Add stock info to holdings
        holdings_df['current_price'] = holdings_df['ticker'].map(
            lambda x: ticker_info.get(x, {}).get('current_price', 0)
        )
        
        # Use Dataroma price if no external price available
        holdings_df['current_price'] = holdings_df.apply(
            lambda row: row['dataroma_price'] if row['current_price'] == 0 and row['dataroma_price'] > 0 else row['current_price'],
            axis=1
        )
        
        holdings_df['market_cap'] = holdings_df['ticker'].map(
            lambda x: ticker_info.get(x, {}).get('market_cap', 0)
        )
        holdings_df['market_cap_category'] = holdings_df['market_cap'].map(self.categorize_market_cap)
        holdings_df['sector'] = holdings_df['ticker'].map(
            lambda x: ticker_info.get(x, {}).get('sector', 'Unknown')
        )
        holdings_df['industry'] = holdings_df['ticker'].map(
            lambda x: ticker_info.get(x, {}).get('industry', 'Unknown')
        )
        
        # Calculate position values
        holdings_df['position_value'] = holdings_df['shares_num'] * holdings_df['current_price']
        
        # Summary of data quality
        print(f"\nData Summary:")
        print(f"- Total holdings: {len(holdings_df)}")
        print(f"- Holdings with portfolio %: {(holdings_df['percent_num'] > 0).sum()}")
        print(f"- Holdings with prices: {(holdings_df['current_price'] > 0).sum()}")
        print(f"- Holdings with market cap: {(holdings_df['market_cap'] > 0).sum()}")
        
        return self.generate_comprehensive_analysis(holdings_df, ticker_info)
    
    def generate_comprehensive_analysis(self, holdings_df, ticker_info):
        """Generate all analysis reports"""
        reports = {}
        
        # Check if we have any price data
        has_price_data = holdings_df['current_price'].sum() > 0
        has_market_cap_data = holdings_df['market_cap'].sum() > 0
        
        # Filter out holdings with invalid portfolio percentages
        valid_holdings = holdings_df[holdings_df['percent_num'] > 0].copy()
        print(f"Analyzing {len(valid_holdings)} holdings with valid portfolio percentages")
        
        # 1. Stocks by price threshold with manager count (only if we have price data)
        if has_price_data:
            price_thresholds = [5, 10, 20, 50, 100, 200, 300]
            for threshold in price_thresholds:
                stocks_under = valid_holdings[
                    (valid_holdings['current_price'] > 0) & 
                    (valid_holdings['current_price'] <= threshold)
                ].copy()
                if not stocks_under.empty:
                    summary = stocks_under.groupby('ticker').agg({
                        'manager': ['count', lambda x: list(x.unique())],
                        'stock': 'first',
                        'current_price': 'first',
                        'percent_num': ['mean', 'max'],
                        'shares_num': 'sum',
                        'market_cap': 'first',
                        'market_cap_category': 'first',
                        'sector': 'first'
                    }).round(2)
                    
                    summary.columns = ['manager_count', 'managers', 'company', 'price', 
                                     'avg_portfolio_pct', 'max_portfolio_pct', 'total_shares',
                                     'market_cap', 'cap_category', 'sector']
                    summary = summary.sort_values('manager_count', ascending=False)
                    
                    reports[f'stocks_under_${threshold}'] = summary
        
        # 2. Conviction Score Analysis (works without price data - MAIN FOCUS)
        if not valid_holdings.empty:
            conviction_df = valid_holdings.groupby('ticker').agg({
                'manager': ['count', lambda x: list(x.unique())],
                'stock': 'first',
                'current_price': 'first',
                'percent_num': ['mean', 'max', 'std'],
                'shares_num': 'sum',
                'market_cap': 'first',
                'market_cap_category': 'first',
                'sector': 'first',
                'industry': 'first'
            }).round(2)
            
            conviction_df.columns = ['manager_count', 'managers', 'company', 'price', 
                                    'avg_portfolio_pct', 'max_portfolio_pct', 'portfolio_pct_std',
                                    'total_shares', 'market_cap', 'cap_category', 'sector', 'industry']
            
            # Calculate conviction score (works without price data)
            conviction_df['conviction_score'] = (
                conviction_df['manager_count'] * 2 +  # Weight manager count heavily
                conviction_df['avg_portfolio_pct'] +
                conviction_df['max_portfolio_pct'] / 2
            ).round(2)
            
            # Ensure non-zero standard deviation
            conviction_df['portfolio_pct_std'] = conviction_df['portfolio_pct_std'].fillna(0)
            
            conviction_df = conviction_df.sort_values('conviction_score', ascending=False)
            reports['high_conviction_stocks'] = conviction_df.head(50)
        else:
            # Use all holdings if no valid portfolio percentages
            conviction_df = holdings_df.groupby('ticker').agg({
                'manager': ['count', lambda x: list(x.unique())],
                'stock': 'first',
                'current_price': 'first',
                'shares_num': 'sum',
                'market_cap': 'first',
                'market_cap_category': 'first',
                'sector': 'first',
                'industry': 'first'
            }).round(2)
            
            conviction_df.columns = ['manager_count', 'managers', 'company', 'price', 
                                    'total_shares', 'market_cap', 'cap_category', 'sector', 'industry']
            
            # Simple conviction score based on manager count only
            conviction_df['conviction_score'] = conviction_df['manager_count'] * 2
            conviction_df['avg_portfolio_pct'] = 0
            conviction_df['max_portfolio_pct'] = 0
            conviction_df['portfolio_pct_std'] = 0
            
            conviction_df = conviction_df.sort_values('manager_count', ascending=False)
            reports['high_conviction_stocks'] = conviction_df.head(50)
        
        # 3. Market Cap Categories (only if we have market cap data)
        if has_market_cap_data:
            for cap_category in ['Micro-Cap', 'Small-Cap', 'Mid-Cap', 'Large-Cap', 'Mega-Cap']:
                cap_stocks = holdings_df[
                    (holdings_df['market_cap_category'] == cap_category) & 
                    (holdings_df['market_cap'] > 0)
                ]
                if not cap_stocks.empty:
                    cap_summary = cap_stocks.groupby('ticker').agg({
                        'manager': ['count', lambda x: list(x.unique())],
                        'stock': 'first',
                        'current_price': 'first',
                        'percent_num': ['mean', 'max'],
                        'market_cap': 'first',
                        'sector': 'first'
                    }).round(2)
                    
                    cap_summary.columns = ['manager_count', 'managers', 'company', 'price',
                                         'avg_portfolio_pct', 'max_portfolio_pct', 'market_cap', 'sector']
                    cap_summary = cap_summary.sort_values('manager_count', ascending=False)
                    
                    reports[f'{cap_category.lower()}_favorites'] = cap_summary.head(20)
        
        # 4. Highest Portfolio Concentration (focus on valid percentages)
        if not valid_holdings.empty:
            high_concentration = valid_holdings.nlargest(50, 'percent_num')[
                ['manager', 'ticker', 'stock', 'percent_num', 'current_price', 
                 'market_cap_category', 'sector']
            ].copy()
        else:
            high_concentration = pd.DataFrame()
        
        if not high_concentration.empty:
            high_concentration.columns = ['manager', 'ticker', 'company', 'portfolio_pct', 
                                         'price', 'cap_category', 'sector']
            reports['highest_portfolio_concentration'] = high_concentration
        
        # 5. Hidden Gems (works without price data)
        if 'high_conviction_stocks' in reports and not reports['high_conviction_stocks'].empty:
            hidden_gems = reports['high_conviction_stocks'][
                (reports['high_conviction_stocks']['manager_count'] <= 3) & 
                (reports['high_conviction_stocks']['avg_portfolio_pct'] >= 3)
            ].head(30)
            reports['hidden_gems'] = hidden_gems
        
        # 6. Sector Analysis (works without price data)
        sector_summary = holdings_df.groupby('sector').agg({
            'ticker': 'nunique',
            'manager': 'count',
            'percent_num': 'mean'
        }).round(2)
        sector_summary.columns = ['unique_stocks', 'total_positions', 'avg_portfolio_pct']
        sector_summary = sector_summary.sort_values('total_positions', ascending=False)
        reports['sector_analysis'] = sector_summary
        
        # 7. Multi-Manager Holdings (works without price data)
        if 'high_conviction_stocks' in reports:
            multi_manager = reports['high_conviction_stocks'][
                reports['high_conviction_stocks']['manager_count'] >= 5
            ].head(30)
            reports['multi_manager_favorites'] = multi_manager
        
        # 8. Concentrated Positions by Manager (works without price data)
        if not valid_holdings.empty:
            concentrated = valid_holdings[valid_holdings['percent_num'] >= 5].copy()
            if not concentrated.empty:
                concentrated_summary = concentrated.groupby('manager').agg({
                    'ticker': ['count', lambda x: list(x)],
                    'percent_num': 'sum'
                }).round(2)
                concentrated_summary.columns = ['concentrated_positions', 'tickers', 'total_portfolio_pct']
                concentrated_summary = concentrated_summary.sort_values('concentrated_positions', ascending=False)
                reports['managers_with_concentrated_bets'] = concentrated_summary.head(20)
        
        # 9. Master Overview
        data_quality = 'Full'
        if not has_price_data and not has_market_cap_data:
            data_quality = 'Dataroma data only (no external prices)'
        elif not has_market_cap_data:
            data_quality = 'Partial (Dataroma prices only)'
        elif not self.yfinance_enabled:
            data_quality = 'Partial (using cached/Dataroma prices)'
        
        # Get top conviction stocks for overview
        top_conviction = {}
        if 'high_conviction_stocks' in reports and not reports['high_conviction_stocks'].empty:
            top_conviction = reports['high_conviction_stocks'].head(10)[
                ['company', 'manager_count', 'avg_portfolio_pct']
            ].to_dict('index')
        
        overview = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'total_managers': holdings_df['manager'].nunique(),
            'total_unique_stocks': holdings_df['ticker'].nunique(),
            'total_positions': len(holdings_df),
            'positions_with_valid_percentages': len(valid_holdings),
            'data_quality': data_quality,
            'market_cap_distribution': holdings_df['market_cap_category'].value_counts().to_dict() if has_market_cap_data else {},
            'sector_distribution': holdings_df['sector'].value_counts().head(10).to_dict(),
            'top_10_most_held': top_conviction,
            'top_5_by_conviction': reports['high_conviction_stocks'].head(5)[
                ['company', 'conviction_score', 'manager_count', 'avg_portfolio_pct']
            ].to_dict('index') if 'high_conviction_stocks' in reports else {},
            'top_5_concentrated_positions': high_concentration.head(5)[
                ['manager', 'company', 'portfolio_pct']
            ].to_dict('index') if not high_concentration.empty else {},
            'top_5_hidden_gems': hidden_gems.head(5)[
                ['company', 'managers', 'avg_portfolio_pct']
            ].to_dict('index') if 'hidden_gems' in reports and not reports['hidden_gems'].empty else {}
        }
        reports['overview'] = overview
        
        return reports
    
    def analyze_changes(self, mode='incremental'):
        """Analyze portfolio changes"""
        if mode == 'incremental':
            return self.analyze_incremental()
        else:
            return self.analyze_full()
    
    def analyze_incremental(self):
        """Analyze only recent changes (fast daily analysis)"""
        print("Running incremental change analysis...")
        
        recent_files = self.get_latest_files(2)
        if len(recent_files) < 2:
            print("Not enough files for incremental analysis")
            return pd.DataFrame()
        
        # Load recent files
        current_df = pd.read_csv(recent_files[0])
        prev_df = pd.read_csv(recent_files[1])
        
        # Extract date from filename
        current_date = os.path.basename(recent_files[0]).replace('holdings_', '').replace('.csv', '')
        current_date = datetime.strptime(current_date, '%Y%m%d').strftime('%Y-%m-%d')
        
        # Analyze changes
        changes = self.compare_holdings(prev_df, current_df, current_date)
        
        # Update cumulative changes file
        self.update_cumulative_changes(changes)
        
        return pd.DataFrame(changes)
    
    def analyze_full(self):
        """Full historical analysis (weekly or on-demand)"""
        print("Running full historical analysis...")
        
        all_files = glob.glob(os.path.join(self.holdings_dir, 'holdings_*.csv'))
        all_files.sort()
        
        all_changes = []
        
        for i in range(1, len(all_files)):
            try:
                prev_df = pd.read_csv(all_files[i-1])
                curr_df = pd.read_csv(all_files[i])
                
                # Extract date from filename
                date = os.path.basename(all_files[i]).replace('holdings_', '').replace('.csv', '')
                date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
                
                changes = self.compare_holdings(prev_df, curr_df, date)
                all_changes.extend(changes)
                
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(all_files)-1} file pairs...")
                    
            except Exception as e:
                print(f"Error processing {all_files[i]}: {e}")
        
        return pd.DataFrame(all_changes)
    
    def compare_holdings(self, prev_df, current_df, date):
        """Smart comparison of holdings"""
        changes = []
        
        # Create unique keys
        prev_df['key'] = prev_df['manager'] + '_' + prev_df['ticker']
        current_df['key'] = current_df['manager'] + '_' + current_df['ticker']
        
        # Convert numeric columns
        for df in [prev_df, current_df]:
            df['shares_num'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)
            df['percent_num'] = pd.to_numeric(df['portfolio_percent'], errors='coerce').fillna(0)
        
        # Find new positions
        new_positions = current_df[~current_df['key'].isin(prev_df['key'])]
        for _, row in new_positions.iterrows():
            if row['shares_num'] > 0:  # Only if we have share data
                stock_info = self.get_stock_info(row['ticker'])
                changes.append({
                    'date': date,
                    'manager': row['manager'],
                    'ticker': row['ticker'],
                    'stock': row['stock'],
                    'action': 'NEW_POSITION',
                    'shares': row['shares_num'],
                    'portfolio_percent': row['percent_num'],
                    'current_price': stock_info.get('current_price', 0),
                    'market_cap': stock_info.get('market_cap', 0),
                    'market_cap_category': self.categorize_market_cap(stock_info.get('market_cap', 0)),
                    'sector': stock_info.get('sector', 'Unknown')
                })
        
        # Find closed positions
        closed_positions = prev_df[~prev_df['key'].isin(current_df['key'])]
        for _, row in closed_positions.iterrows():
            if row['shares_num'] > 0:
                stock_info = self.get_stock_info(row['ticker'])
                changes.append({
                    'date': date,
                    'manager': row['manager'],
                    'ticker': row['ticker'],
                    'stock': row['stock'],
                    'action': 'CLOSED_POSITION',
                    'shares': row['shares_num'],
                    'portfolio_percent': 0,
                    'current_price': stock_info.get('current_price', 0),
                    'market_cap': stock_info.get('market_cap', 0),
                    'market_cap_category': self.categorize_market_cap(stock_info.get('market_cap', 0)),
                    'sector': stock_info.get('sector', 'Unknown')
                })
        
        # Find position changes
        for key in set(prev_df['key']) & set(current_df['key']):
            prev_row = prev_df[prev_df['key'] == key].iloc[0]
            curr_row = current_df[current_df['key'] == key].iloc[0]
            
            share_change = curr_row['shares_num'] - prev_row['shares_num']
            percent_change = curr_row['percent_num'] - prev_row['percent_num']
            
            # Significant change in shares or portfolio percentage
            if abs(share_change) > 100 or abs(percent_change) > 0.5:
                stock_info = self.get_stock_info(curr_row['ticker'])
                changes.append({
                    'date': date,
                    'manager': curr_row['manager'],
                    'ticker': curr_row['ticker'],
                    'stock': curr_row['stock'],
                    'action': 'INCREASED' if share_change > 0 else 'DECREASED',
                    'shares': abs(share_change),
                    'portfolio_percent': curr_row['percent_num'],
                    'portfolio_percent_change': percent_change,
                    'current_price': stock_info.get('current_price', 0),
                    'market_cap': stock_info.get('market_cap', 0),
                    'market_cap_category': self.categorize_market_cap(stock_info.get('market_cap', 0)),
                    'sector': stock_info.get('sector', 'Unknown')
                })
        
        return changes
    
    def update_cumulative_changes(self, new_changes):
        """Update the cumulative changes file efficiently"""
        cumulative_file = os.path.join(self.analysis_dir, 'all_changes.csv')
        
        if os.path.exists(cumulative_file):
            existing_df = pd.read_csv(cumulative_file)
            new_df = pd.DataFrame(new_changes)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on key columns
            combined_df = combined_df.drop_duplicates(
                subset=['date', 'manager', 'ticker', 'action'], 
                keep='last'
            )
        else:
            combined_df = pd.DataFrame(new_changes)
        
        combined_df.to_csv(cumulative_file, index=False)
    
    def save_all_reports(self, reports):
        """Save all analysis reports"""
        print("\nSaving analysis reports...")
        
        # Create master stock analysis folder
        master_folder = os.path.join(self.analysis_dir, 'master_stock_analysis')
        
        # Save overview as JSON
        if 'overview' in reports:
            with open(os.path.join(self.analysis_dir, 'overview.json'), 'w') as f:
                json.dump(reports['overview'], f, indent=2, default=str)
            # Also save in master folder
            with open(os.path.join(master_folder, 'overview.json'), 'w') as f:
                json.dump(reports['overview'], f, indent=2, default=str)
        
        # Define organized report groups for the master folder
        report_groups = {
            'price_thresholds': [
                'stocks_under_$5', 'stocks_under_$10', 'stocks_under_$20',
                'stocks_under_$50', 'stocks_under_$100', 'stocks_under_$200',
                'stocks_under_$300'
            ],
            'market_cap_analysis': [
                'micro-cap_favorites', 'small-cap_favorites', 'mid-cap_favorites',
                'large-cap_favorites', 'mega-cap_favorites'
            ],
            'conviction_analysis': [
                'high_conviction_stocks', 'hidden_gems', 
                'highest_portfolio_concentration', 'multi_manager_favorites',
                'managers_with_concentrated_bets'
            ],
            'sector_analysis': ['sector_analysis'],
            'recent_activity': ['recent_accumulation']
        }
        
        # Save each report as CSV in both locations
        for report_name, report_data in reports.items():
            if report_name == 'overview':
                continue
                
            if isinstance(report_data, pd.DataFrame) and not report_data.empty:
                # Convert list columns to strings for CSV
                df = report_data.copy()
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else x
                        )
                
                # Save in main analysis folder
                filename = os.path.join(self.analysis_dir, f'{report_name}.csv')
                df.to_csv(filename)
                
                # Save in master folder with organized structure
                master_filename = os.path.join(master_folder, f'{report_name}.csv')
                df.to_csv(master_filename)
                
                print(f"  Saved {report_name}.csv ({len(df)} rows)")
        
        # Create a master Excel file with all reports
        excel_file = os.path.join(self.analysis_dir, 'dataroma_analysis.xlsx')
        master_excel_file = os.path.join(master_folder, 'dataroma_complete_analysis.xlsx')
        
        # Create the Excel file
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Write overview
            overview_df = pd.DataFrame([reports.get('overview', {})])
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Write other reports
            sheet_order = [
                'high_conviction_stocks',
                'multi_manager_favorites',
                'stocks_under_$10',
                'stocks_under_$20',
                'stocks_under_$50',
                'micro-cap_favorites',
                'small-cap_favorites',
                'mid-cap_favorites',
                'hidden_gems',
                'highest_portfolio_concentration',
                'managers_with_concentrated_bets',
                'sector_analysis'
            ]
            
            for sheet_name in sheet_order:
                if sheet_name in reports and isinstance(reports[sheet_name], pd.DataFrame):
                    df = reports[sheet_name].copy()
                    # Convert lists to strings
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].apply(
                                lambda x: ', '.join(x) if isinstance(x, list) else x
                            )
                    df.to_excel(writer, sheet_name=sheet_name[:31])  # Excel sheet name limit
        
        # Copy Excel file to master folder
        import shutil
        shutil.copy2(excel_file, master_excel_file)
        
        # Create README for master folder
        readme_content = """# Dataroma Stock Analysis Reports

This folder contains individual CSV files extracted from the complete Dataroma analysis.

## Files Included:

### Overview
- `overview.json` - Summary statistics and key findings

### Price Threshold Analysis (when price data available)
- `stocks_under_$5.csv` - Stocks trading under $5
- `stocks_under_$10.csv` - Stocks trading under $10
- `stocks_under_$20.csv` - Stocks trading under $20
- `stocks_under_$50.csv` - Stocks trading under $50
- `stocks_under_$100.csv` - Stocks trading under $100
- `stocks_under_$200.csv` - Stocks trading under $200
- `stocks_under_$300.csv` - Stocks trading under $300

### Market Cap Analysis (when market cap data available)
- `micro-cap_favorites.csv` - Top micro-cap stocks (<$300M)
- `small-cap_favorites.csv` - Top small-cap stocks ($300M-$2B)
- `mid-cap_favorites.csv` - Top mid-cap stocks ($2B-$10B)
- `large-cap_favorites.csv` - Top large-cap stocks ($10B-$200B)
- `mega-cap_favorites.csv` - Top mega-cap stocks (>$200B)

### Conviction Analysis (always available)
- `high_conviction_stocks.csv` - Stocks with highest conviction scores
- `hidden_gems.csv` - Under-the-radar high-conviction picks
- `highest_portfolio_concentration.csv` - Largest individual positions
- `multi_manager_favorites.csv` - Stocks held by 5+ managers
- `managers_with_concentrated_bets.csv` - Managers with concentrated positions

### Other Analysis
- `sector_analysis.csv` - Breakdown by sector
- `recent_accumulation.csv` - Recent buying activity (if available)

### Complete Analysis
- `dataroma_complete_analysis.xlsx` - All reports in one Excel file

## Column Descriptions:
- `manager_count` - Number of superinvestors holding the stock
- `managers` - List of managers holding the stock
- `avg_portfolio_pct` - Average percentage of portfolio
- `max_portfolio_pct` - Maximum percentage in any portfolio
- `conviction_score` - Combined score based on manager count and portfolio weight
- `market_cap` - Current market capitalization (when available)
- `cap_category` - Market cap category (when available)
- `price` - Current stock price (when available)

## Data Sources:
By default, this analysis uses only data available from Dataroma to ensure reliability.
External price and market cap data can be enabled with the --enable-yfinance flag,
but this may cause errors due to API limitations.

## Usage:
- For daily analysis: `python analyze_holdings.py --mode current`
- For change tracking: `python analyze_holdings.py --mode current --changes`
- With external data: `python analyze_holdings.py --mode current --enable-yfinance`
"""
        
        readme_file = os.path.join(master_folder, 'README.md')
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"\nMaster Excel file saved: {excel_file}")
        print(f"Individual CSV files saved in: {master_folder}")
        
        # Summary of what was generated
        print(f"\nReports generated:")
        price_reports = [k for k in reports.keys() if k.startswith('stocks_under_')]
        cap_reports = [k for k in reports.keys() if k.endswith('_favorites') and 'cap' in k]
        
        if price_reports:
            print(f"- Price threshold reports: {len(price_reports)}")
        if cap_reports:
            print(f"- Market cap category reports: {len(cap_reports)}")
        
        always_available = ['high_conviction_stocks', 'multi_manager_favorites', 
                          'highest_portfolio_concentration', 'sector_analysis']
        available_count = sum(1 for r in always_available if r in reports and not reports[r].empty)
        print(f"- Core analysis reports: {available_count}/{len(always_available)}")
        
        # Save caches
        self.save_caches()

def main():
    parser = argparse.ArgumentParser(description='Analyze Dataroma holdings')
    parser.add_argument('--mode', choices=['incremental', 'full', 'current'], 
                       default='current', 
                       help='Analysis mode: current (analyze current holdings), incremental (recent changes), or full (all historical changes)')
    parser.add_argument('--changes', action='store_true',
                       help='Also analyze portfolio changes')
    parser.add_argument('--enable-yfinance', action='store_true',
                       help='Enable external price/market cap fetching via yfinance (may cause errors)')
    args = parser.parse_args()
    
    analyzer = SmartHoldingsAnalyzer()
    
    # Only enable yfinance if explicitly requested
    if args.enable_yfinance:
        analyzer.yfinance_enabled = True
        print("External data fetching enabled (--enable-yfinance flag set)")
        print("Warning: This may cause HTTP errors and slow performance")
    else:
        print("Running with Dataroma data only (recommended)")
    
    # Always analyze current holdings
    print("=" * 50)
    print("DATAROMA HOLDINGS ANALYZER")
    print("=" * 50)
    
    current_reports = analyzer.analyze_current_holdings()
    
    # Optionally analyze changes
    if args.changes or args.mode != 'current':
        if args.mode == 'incremental':
            changes_df = analyzer.analyze_incremental()
        elif args.mode == 'full':
            changes_df = analyzer.analyze_full()
        
        if args.changes and not changes_df.empty:
            print(f"\nAnalyzed {len(changes_df)} portfolio changes")
            # Add change analysis to reports
            recent_changes = changes_df[changes_df['action'].isin(['NEW_POSITION', 'INCREASED'])]
            if not recent_changes.empty:
                change_summary = recent_changes.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())],
                    'stock': 'first',
                    'current_price': 'first',
                    'market_cap_category': 'first'
                })
                change_summary.columns = ['managers_buying', 'managers', 'company', 'price', 'cap_category']
                current_reports['recent_accumulation'] = change_summary.sort_values('managers_buying', ascending=False).head(30)
    
    # Save all reports
    analyzer.save_all_reports(current_reports)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f"\nKey findings:")
    overview = current_reports.get('overview', {})
    print(f"- Total managers tracked: {overview.get('total_managers', 0)}")
    print(f"- Total unique stocks: {overview.get('total_unique_stocks', 0)}")
    print(f"- Total positions: {overview.get('total_positions', 0)}")
    if 'positions_with_valid_percentages' in overview:
        print(f"- Positions with portfolio %: {overview.get('positions_with_valid_percentages', 0)}")
    print(f"- Data quality: {overview.get('data_quality', 'Unknown')}")
    
    print(f"\nAnalysis files saved in:")
    print(f"- Main folder: {analyzer.analysis_dir}/")
    print(f"- Individual CSVs: {analyzer.analysis_dir}/master_stock_analysis/")
    
    if 'high_conviction_stocks' in current_reports and not current_reports['high_conviction_stocks'].empty:
        top_conviction = current_reports['high_conviction_stocks'].head(5)
        print(f"\nTop 5 High Conviction Stocks:")
        for ticker, row in top_conviction.iterrows():
            output = f"  {ticker}: {row['company']} - {row['manager_count']} managers"
            
            # Add portfolio percentage if available
            if row['avg_portfolio_pct'] > 0:
                output += f", {row['avg_portfolio_pct']:.1f}% avg position"
            
            # Add price if available
            if row['price'] > 0:
                output += f", ${row['price']:.2f}"
                
            print(output)

if __name__ == "__main__":
    main()