import pandas as pd
import git
import os
import yfinance as yf
from datetime import datetime, timedelta
import json
import argparse
import glob

class SmartHoldingsAnalyzer:
    def __init__(self):
        self.repo = git.Repo('.')
        self.price_cache = {}
        self.holdings_dir = "holdings"
        self.analysis_dir = "analysis"
        self.load_price_cache()
    
    def load_price_cache(self):
        """Load cached prices to avoid repeated API calls"""
        cache_file = os.path.join(self.analysis_dir, 'price_cache.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.price_cache = json.load(f)
    
    def save_price_cache(self):
        """Save price cache for future use"""
        cache_file = os.path.join(self.analysis_dir, 'price_cache.json')
        with open(cache_file, 'w') as f:
            json.dump(self.price_cache, f)
    
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
    
    def analyze_incremental(self):
        """Analyze only recent changes (fast daily analysis)"""
        print("Running incremental analysis...")
        
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
            df['shares_num'] = pd.to_numeric(df['shares'], errors='coerce')
            df['percent_num'] = pd.to_numeric(df['portfolio_percent'], errors='coerce')
        
        # Find new positions
        new_positions = current_df[~current_df['key'].isin(prev_df['key'])]
        for _, row in new_positions.iterrows():
            price = self.get_historical_price(row['ticker'], date)
            if price:  # Only add if we have price data
                changes.append({
                    'date': date,
                    'manager': row['manager'],
                    'ticker': row['ticker'],
                    'stock': row['stock'],
                    'action': 'BUY',
                    'shares': row['shares_num'],
                    'price': price,
                    'portfolio_percent': row['percent_num'],
                    'value': row['shares_num'] * price if row['shares_num'] else None
                })
        
        # Find closed positions
        closed_positions = prev_df[~prev_df['key'].isin(current_df['key'])]
        for _, row in closed_positions.iterrows():
            price = self.get_historical_price(row['ticker'], date)
            if price:
                changes.append({
                    'date': date,
                    'manager': row['manager'],
                    'ticker': row['ticker'],
                    'stock': row['stock'],
                    'action': 'SELL',
                    'shares': row['shares_num'],
                    'price': price,
                    'portfolio_percent': 0,
                    'value': row['shares_num'] * price if row['shares_num'] else None
                })
        
        # Find position changes
        for key in set(prev_df['key']) & set(current_df['key']):
            prev_row = prev_df[prev_df['key'] == key].iloc[0]
            curr_row = current_df[current_df['key'] == key].iloc[0]
            
            share_change = curr_row['shares_num'] - prev_row['shares_num']
            if abs(share_change) > 100:  # Significant change
                price = self.get_historical_price(curr_row['ticker'], date)
                if price:
                    changes.append({
                        'date': date,
                        'manager': curr_row['manager'],
                        'ticker': curr_row['ticker'],
                        'stock': curr_row['stock'],
                        'action': 'INCREASE' if share_change > 0 else 'DECREASE',
                        'shares': abs(share_change),
                        'price': price,
                        'portfolio_percent': curr_row['percent_num'],
                        'value': abs(share_change) * price
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
    
    def generate_analysis_reports(self, changes_df):
        """Generate comprehensive analysis reports"""
        if changes_df.empty:
            return {}
        
        reports = {}
        
        # Filter for valid prices
        changes_df = changes_df[changes_df['price'].notna()]
        
        # Stocks under different price thresholds
        buys_df = changes_df[changes_df['action'] == 'BUY'].copy()
        
        for threshold in [5, 10, 20, 50]:
            threshold_df = buys_df[buys_df['price'] < threshold]
            if not threshold_df.empty:
                summary = threshold_df.groupby('ticker').agg({
                    'manager': 'count',
                    'stock': 'first',
                    'price': 'mean',
                    'shares': 'sum'
                }).rename(columns={'manager': 'buy_count', 'price': 'avg_price'})
                summary = summary.sort_values('buy_count', ascending=False)
                reports[f'acquisitions_under_${threshold}'] = summary.to_dict('index')
        
        # Most bought stocks (by number of managers)
        buy_summary = buys_df.groupby('ticker').agg({
            'manager': lambda x: list(x.unique()),
            'stock': 'first',
            'shares': 'sum',
            'value': 'sum'
        }).rename(columns={'manager': 'managers'})
        buy_summary['manager_count'] = buy_summary['managers'].apply(len)
        buy_summary = buy_summary.sort_values('manager_count', ascending=False).head(30)
        reports['most_bought_stocks'] = buy_summary.to_dict('index')
        
        # Most sold stocks
        sells_df = changes_df[changes_df['action'] == 'SELL']
        if not sells_df.empty:
            sell_summary = sells_df.groupby('ticker').agg({
                'manager': lambda x: list(x.unique()),
                'stock': 'first',
                'shares': 'sum',
                'value': 'sum'
            }).rename(columns={'manager': 'managers'})
            sell_summary['manager_count'] = sell_summary['managers'].apply(len)
            sell_summary = sell_summary.sort_values('manager_count', ascending=False).head(30)
            reports['most_sold_stocks'] = sell_summary.to_dict('index')
        
        # Recent activity summary
        if 'date' in changes_df.columns:
            recent_date = changes_df['date'].max()
            recent_changes = changes_df[changes_df['date'] == recent_date]
            
            reports['recent_activity'] = {
                'date': recent_date,
                'total_changes': len(recent_changes),
                'buys': len(recent_changes[recent_changes['action'] == 'BUY']),
                'sells': len(recent_changes[recent_changes['action'] == 'SELL']),
                'increases': len(recent_changes[recent_changes['action'] == 'INCREASE']),
                'decreases': len(recent_changes[recent_changes['action'] == 'DECREASE']),
                'total_buy_value': recent_changes[recent_changes['action'] == 'BUY']['value'].sum(),
                'total_sell_value': recent_changes[recent_changes['action'] == 'SELL']['value'].sum()
            }
        
        return reports
    
    def save_reports(self, reports):
        """Save analysis reports efficiently"""
        # Save main summary
        with open(os.path.join(self.analysis_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(reports, f, indent=2, default=str)
        
        # Save individual CSV reports
        for threshold in [5, 10, 20, 50]:
            key = f'acquisitions_under_${threshold}'
            if key in reports and reports[key]:
                df = pd.DataFrame.from_dict(reports[key], orient='index')
                df.index.name = 'ticker'
                df.to_csv(os.path.join(self.analysis_dir, f'{key}.csv'))
        
        # Save top picks
        for report_type in ['most_bought_stocks', 'most_sold_stocks']:
            if report_type in reports and reports[report_type]:
                df = pd.DataFrame.from_dict(reports[report_type], orient='index')
                # Convert manager lists to string for CSV
                if 'managers' in df.columns:
                    df['managers'] = df['managers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                df.index.name = 'ticker'
                df.to_csv(os.path.join(self.analysis_dir, f'{report_type}.csv'))
        
        # Save price cache
        self.save_price_cache()

def main():
    parser = argparse.ArgumentParser(description='Analyze Dataroma holdings')
    parser.add_argument('--mode', choices=['incremental', 'full'], 
                       default='incremental', 
                       help='Analysis mode: incremental (fast) or full (comprehensive)')
    args = parser.parse_args()
    
    analyzer = SmartHoldingsAnalyzer()
    
    if args.mode == 'incremental':
        changes_df = analyzer.analyze_incremental()
    else:
        changes_df = analyzer.analyze_full()
    
    if not changes_df.empty:
        print(f"Analyzed {len(changes_df)} changes")
        reports = analyzer.generate_analysis_reports(changes_df)
        analyzer.save_reports(reports)
        print("Analysis complete!")
    else:
        print("No changes to analyze")

if __name__ == "__main__":
    main()