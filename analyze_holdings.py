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

class SmartHoldingsAnalyzer:
    def __init__(self):
        self.repo = git.Repo('.')
        self.price_cache = {}
        self.market_cap_cache = {}
        self.holdings_dir = "holdings"
        self.analysis_dir = "analysis"
        self.ensure_directories()
        self.load_caches()
    
    def ensure_directories(self):
        """Ensure analysis directory exists"""
        os.makedirs(self.analysis_dir, exist_ok=True)
    
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
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            stock_data = {
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'company_name': info.get('longName', info.get('shortName', ticker)),
                'timestamp': datetime.now().isoformat()
            }
            
            self.market_cap_cache[cache_key] = stock_data
            return stock_data
        except:
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
        
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=7)
            
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self.price_cache[cache_key] = price
                return price
        except:
            pass
        
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
        
        # Convert numeric columns
        holdings_df['shares_num'] = pd.to_numeric(holdings_df['shares'], errors='coerce')
        holdings_df['percent_num'] = pd.to_numeric(holdings_df['portfolio_percent'], errors='coerce')
        
        # Get stock info for all unique tickers
        print("Fetching stock information...")
        ticker_info = {}
        unique_tickers = holdings_df['ticker'].unique()
        
        for i, ticker in enumerate(unique_tickers):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(unique_tickers)} tickers...")
            ticker_info[ticker] = self.get_stock_info(ticker)
        
        # Add stock info to holdings
        holdings_df['current_price'] = holdings_df['ticker'].map(
            lambda x: ticker_info.get(x, {}).get('current_price', 0)
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
        
        return self.generate_comprehensive_analysis(holdings_df, ticker_info)
    
    def generate_comprehensive_analysis(self, holdings_df, ticker_info):
        """Generate all analysis reports"""
        reports = {}
        
        # 1. Stocks by price threshold with manager count
        price_thresholds = [5, 10, 20, 50, 100, 200, 300]
        for threshold in price_thresholds:
            stocks_under = holdings_df[holdings_df['current_price'] <= threshold].copy()
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
        
        # 2. Conviction Score Analysis (multiple managers + high portfolio %)
        conviction_df = holdings_df.groupby('ticker').agg({
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
        
        # Calculate conviction score
        conviction_df['conviction_score'] = (
            conviction_df['manager_count'] * 2 +  # Weight manager count heavily
            conviction_df['avg_portfolio_pct'] +
            conviction_df['max_portfolio_pct'] / 2
        ).round(2)
        
        conviction_df = conviction_df.sort_values('conviction_score', ascending=False)
        reports['high_conviction_stocks'] = conviction_df.head(50)
        
        # 3. Market Cap Categories with Top Picks
        for cap_category in ['Micro-Cap', 'Small-Cap', 'Mid-Cap', 'Large-Cap', 'Mega-Cap']:
            cap_stocks = holdings_df[holdings_df['market_cap_category'] == cap_category]
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
        
        # 4. Highest Portfolio Concentration (Top positions by % of portfolio)
        high_concentration = holdings_df.nlargest(50, 'percent_num')[
            ['manager', 'ticker', 'stock', 'percent_num', 'current_price', 
             'market_cap_category', 'sector']
        ].copy()
        high_concentration.columns = ['manager', 'ticker', 'company', 'portfolio_pct', 
                                     'price', 'cap_category', 'sector']
        reports['highest_portfolio_concentration'] = high_concentration
        
        # 5. Hidden Gems (Low manager count but high portfolio % when held)
        hidden_gems = conviction_df[
            (conviction_df['manager_count'] <= 3) & 
            (conviction_df['avg_portfolio_pct'] >= 3)
        ].head(30)
        reports['hidden_gems'] = hidden_gems
        
        # 6. Sector Analysis
        sector_summary = holdings_df.groupby('sector').agg({
            'ticker': 'nunique',
            'manager': 'count',
            'percent_num': 'mean'
        }).round(2)
        sector_summary.columns = ['unique_stocks', 'total_positions', 'avg_portfolio_pct']
        sector_summary = sector_summary.sort_values('total_positions', ascending=False)
        reports['sector_analysis'] = sector_summary
        
        # 7. Master Overview
        overview = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'total_managers': holdings_df['manager'].nunique(),
            'total_unique_stocks': holdings_df['ticker'].nunique(),
            'total_positions': len(holdings_df),
            'market_cap_distribution': holdings_df['market_cap_category'].value_counts().to_dict(),
            'sector_distribution': holdings_df['sector'].value_counts().head(10).to_dict(),
            'top_10_most_held': conviction_df.head(10)[['company', 'manager_count', 'price', 'cap_category']].to_dict('index'),
            'top_5_by_conviction': conviction_df.head(5)[['company', 'conviction_score', 'manager_count', 'avg_portfolio_pct']].to_dict('index'),
            'top_5_micro_caps': reports.get('micro-cap_favorites', pd.DataFrame()).head(5)[['company', 'manager_count', 'price']].to_dict('index') if 'micro-cap_favorites' in reports else {},
            'top_5_hidden_gems': hidden_gems.head(5)[['company', 'managers', 'avg_portfolio_pct', 'price']].to_dict('index') if not hidden_gems.empty else {}
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
        
        # Save overview as JSON
        if 'overview' in reports:
            with open(os.path.join(self.analysis_dir, 'overview.json'), 'w') as f:
                json.dump(reports['overview'], f, indent=2, default=str)
        
        # Save each report as CSV
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
                
                filename = os.path.join(self.analysis_dir, f'{report_name}.csv')
                df.to_csv(filename)
                print(f"  Saved {report_name}.csv ({len(df)} rows)")
        
        # Create a master Excel file with all reports
        excel_file = os.path.join(self.analysis_dir, 'dataroma_analysis.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Write overview
            overview_df = pd.DataFrame([reports.get('overview', {})])
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Write other reports
            sheet_order = [
                'high_conviction_stocks',
                'stocks_under_$10',
                'stocks_under_$20',
                'stocks_under_$50',
                'micro-cap_favorites',
                'small-cap_favorites',
                'mid-cap_favorites',
                'hidden_gems',
                'highest_portfolio_concentration',
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
        
        print(f"\nMaster Excel file saved: {excel_file}")
        
        # Save caches
        self.save_caches()

def main():
    parser = argparse.ArgumentParser(description='Analyze Dataroma holdings')
    parser.add_argument('--mode', choices=['incremental', 'full', 'current'], 
                       default='current', 
                       help='Analysis mode: current (analyze current holdings), incremental (recent changes), or full (all historical changes)')
    parser.add_argument('--changes', action='store_true',
                       help='Also analyze portfolio changes')
    args = parser.parse_args()
    
    analyzer = SmartHoldingsAnalyzer()
    
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
    
    if 'high_conviction_stocks' in current_reports:
        top_conviction = current_reports['high_conviction_stocks'].head(5)
        print(f"\nTop 5 High Conviction Stocks:")
        for ticker, row in top_conviction.iterrows():
            print(f"  {ticker}: {row['company']} - {row['manager_count']} managers, "
                  f"{row['avg_portfolio_pct']:.1f}% avg position, ${row['price']:.2f}")

if __name__ == "__main__":
    main()