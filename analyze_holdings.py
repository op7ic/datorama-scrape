import pandas as pd
import os
from datetime import datetime
import json
import argparse
import glob
import numpy as np
from collections import defaultdict
import shutil

class DataromaAnalyzer:
    def __init__(self):
        self.holdings_dir = "holdings"
        self.analysis_dir = "analysis"
        self.cache_dir = "cache"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all directories exist"""
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(os.path.join(self.analysis_dir, 'master_stock_analysis'), exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
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
    
    def get_latest_files(self, n=2):
        """Get the n most recent holdings files"""
        files = glob.glob(os.path.join(self.holdings_dir, 'holdings_*.csv'))
        files.sort(reverse=True)
        return files[:n]
    
    def analyze_current_holdings(self):
        """Analyze current holdings using only Dataroma data"""
        print("Analyzing current holdings...")
        
        # Get latest holdings file
        latest_files = self.get_latest_files(1)
        if not latest_files:
            print("No holdings files found")
            return {}
        
        holdings_df = pd.read_csv(latest_files[0])
        print(f"Loaded {len(holdings_df)} holdings from {os.path.basename(latest_files[0])}")
        
        # Convert numeric columns
        numeric_columns = {
            'shares': 'shares_num',
            'portfolio_percent': 'percent_num',
            'current_price': 'current_price',
            'reported_price': 'reported_price',
            'market_cap': 'market_cap',
            '52_week_high': '52_week_high',
            '52_week_low': '52_week_low',
            'pe_ratio': 'pe_ratio',
            'dividend_yield': 'dividend_yield'
        }
        
        for col, new_col in numeric_columns.items():
            if col in holdings_df.columns:
                if col == 'shares':
                    # Handle shares with commas
                    holdings_df[new_col] = pd.to_numeric(
                        holdings_df[col].astype(str).str.replace(',', '').str.strip(), 
                        errors='coerce'
                    ).fillna(0)
                else:
                    holdings_df[new_col] = pd.to_numeric(holdings_df[col], errors='coerce').fillna(0)
            else:
                holdings_df[new_col] = 0
        
        # Add market cap category
        holdings_df['market_cap_category'] = holdings_df['market_cap'].map(self.categorize_market_cap)
        
        # Ensure sector column exists
        if 'sector' not in holdings_df.columns:
            holdings_df['sector'] = 'Unknown'
        
        # Calculate gain/loss if we have both prices
        holdings_df['gain_loss_pct'] = 0
        mask = (holdings_df['reported_price'] > 0) & (holdings_df['current_price'] > 0)
        holdings_df.loc[mask, 'gain_loss_pct'] = (
            (holdings_df.loc[mask, 'current_price'] - holdings_df.loc[mask, 'reported_price']) / 
            holdings_df.loc[mask, 'reported_price'] * 100
        ).round(2)
        
        # Calculate position values
        holdings_df['position_value'] = holdings_df['shares_num'] * holdings_df['current_price']
        
        # Summary
        print(f"\nData Summary:")
        print(f"- Total holdings: {len(holdings_df)}")
        print(f"- Holdings with portfolio %: {(holdings_df['percent_num'] > 0).sum()}")
        print(f"- Holdings with current price: {(holdings_df['current_price'] > 0).sum()}")
        print(f"- Holdings with market cap: {(holdings_df['market_cap'] > 0).sum()}")
        print(f"- Holdings with sector: {(holdings_df['sector'] != 'Unknown').sum()}")
        print(f"- Holdings with gain/loss: {(holdings_df['gain_loss_pct'] != 0).sum()}")
        
        return self.generate_all_reports(holdings_df)
    
    def generate_all_reports(self, holdings_df):
        """Generate all analysis reports"""
        reports = {}
        
        # 1. Price Threshold Analysis
        price_thresholds = [5, 10, 20, 50, 100, 200, 300]
        for threshold in price_thresholds:
            stocks_under = holdings_df[
                (holdings_df['current_price'] > 0) & 
                (holdings_df['current_price'] <= threshold)
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
                    'sector': 'first',
                    'gain_loss_pct': 'mean'
                }).round(2)
                
                summary.columns = ['manager_count', 'managers', 'company', 'price', 
                                 'avg_portfolio_pct', 'max_portfolio_pct', 'total_shares',
                                 'market_cap', 'cap_category', 'sector', 'avg_gain_loss_pct']
                summary = summary.sort_values('manager_count', ascending=False)
                
                reports[f'stocks_under_${threshold}'] = summary
            else:
                # Create empty dataframe with correct structure
                reports[f'stocks_under_${threshold}'] = pd.DataFrame(columns=[
                    'manager_count', 'managers', 'company', 'price', 
                    'avg_portfolio_pct', 'max_portfolio_pct', 'total_shares',
                    'market_cap', 'cap_category', 'sector', 'avg_gain_loss_pct'
                ])
        
        # 2. Conviction Score Analysis
        conviction_df = holdings_df.groupby('ticker').agg({
            'manager': ['count', lambda x: list(x.unique())],
            'stock': 'first',
            'current_price': 'first',
            'percent_num': ['mean', 'max', 'std'],
            'shares_num': 'sum',
            'market_cap': 'first',
            'market_cap_category': 'first',
            'sector': 'first',
            'industry': 'first' if 'industry' in holdings_df.columns else lambda x: 'Unknown'
        }).round(2)
        
        conviction_df.columns = ['manager_count', 'managers', 'company', 'price', 
                                'avg_portfolio_pct', 'max_portfolio_pct', 'portfolio_pct_std',
                                'total_shares', 'market_cap', 'cap_category', 'sector', 'industry']
        
        # Calculate conviction score
        conviction_df['conviction_score'] = (
            conviction_df['manager_count'] * 2 +
            conviction_df['avg_portfolio_pct'] +
            conviction_df['max_portfolio_pct'] / 2
        ).round(2)
        
        conviction_df['portfolio_pct_std'] = conviction_df['portfolio_pct_std'].fillna(0)
        conviction_df = conviction_df.sort_values('conviction_score', ascending=False)
        reports['high_conviction_stocks'] = conviction_df.head(50)
        
        # 3. Market Cap Categories
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
            else:
                reports[f'{cap_category.lower()}_favorites'] = pd.DataFrame()
        
        # 4. Highest Portfolio Concentration
        high_concentration = holdings_df[holdings_df['percent_num'] > 0].nlargest(50, 'percent_num')[
            ['manager', 'ticker', 'stock', 'percent_num', 'current_price', 
             'market_cap_category', 'sector', 'gain_loss_pct']
        ].copy()
        
        if not high_concentration.empty:
            high_concentration.columns = ['manager', 'ticker', 'company', 'portfolio_pct', 
                                         'price', 'cap_category', 'sector', 'gain_loss_pct']
            reports['highest_portfolio_concentration'] = high_concentration
        else:
            reports['highest_portfolio_concentration'] = pd.DataFrame()
        
        # 5. Hidden Gems
        hidden_gems_mask = (conviction_df['manager_count'] <= 3) & (conviction_df['avg_portfolio_pct'] >= 3)
        if hidden_gems_mask.any():
            hidden_gems = conviction_df[hidden_gems_mask].head(30)
        else:
            # Alternative: stocks with only 1-2 managers but some portfolio allocation
            hidden_gems = conviction_df[
                (conviction_df['manager_count'] <= 2) & 
                (conviction_df['avg_portfolio_pct'] > 0)
            ].head(30)
        
        reports['hidden_gems'] = hidden_gems
        
        # 6. Sector Analysis
        sector_summary = holdings_df.groupby('sector').agg({
            'ticker': 'nunique',
            'manager': 'count',
            'percent_num': 'mean',
            'position_value': 'sum'
        }).round(2)
        
        sector_summary.columns = ['unique_stocks', 'total_positions', 'avg_portfolio_pct', 'total_value']
        sector_summary = sector_summary.sort_values('total_positions', ascending=False)
        reports['sector_analysis'] = sector_summary
        
        # 7. Multi-Manager Holdings
        multi_manager = conviction_df[conviction_df['manager_count'] >= 5].head(30)
        if multi_manager.empty:
            multi_manager = conviction_df[conviction_df['manager_count'] >= 3].head(30)
        reports['multi_manager_favorites'] = multi_manager
        
        # 8. Concentrated Positions by Manager
        concentrated = holdings_df[holdings_df['percent_num'] >= 5].copy()
        if not concentrated.empty:
            concentrated_summary = concentrated.groupby('manager').agg({
                'ticker': ['count', lambda x: list(x)],
                'percent_num': 'sum'
            }).round(2)
            concentrated_summary.columns = ['concentrated_positions', 'tickers', 'total_portfolio_pct']
            concentrated_summary = concentrated_summary.sort_values('concentrated_positions', ascending=False)
            reports['managers_with_concentrated_bets'] = concentrated_summary.head(20)
        else:
            reports['managers_with_concentrated_bets'] = pd.DataFrame()
        
        # 9. Gainers and Losers
        if (holdings_df['gain_loss_pct'] != 0).any():
            # Top Gainers
            gainers = holdings_df[holdings_df['gain_loss_pct'] > 0].copy()
            if not gainers.empty:
                top_gainers = gainers.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())],
                    'stock': 'first',
                    'reported_price': 'mean',
                    'current_price': 'first',
                    'gain_loss_pct': 'mean',
                    'percent_num': 'mean'
                }).round(2)
                
                top_gainers.columns = ['manager_count', 'managers', 'company', 
                                      'avg_reported_price', 'current_price', 
                                      'avg_gain_pct', 'avg_portfolio_pct']
                top_gainers = top_gainers.sort_values('avg_gain_pct', ascending=False)
                reports['top_gainers'] = top_gainers.head(30)
            
            # Top Losers
            losers = holdings_df[holdings_df['gain_loss_pct'] < 0].copy()
            if not losers.empty:
                top_losers = losers.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())],
                    'stock': 'first',
                    'reported_price': 'mean',
                    'current_price': 'first',
                    'gain_loss_pct': 'mean',
                    'percent_num': 'mean'
                }).round(2)
                
                top_losers.columns = ['manager_count', 'managers', 'company', 
                                     'avg_reported_price', 'current_price', 
                                     'avg_loss_pct', 'avg_portfolio_pct']
                top_losers = top_losers.sort_values('avg_loss_pct', ascending=True)
                reports['top_losers'] = top_losers.head(30)
        
        # 10. Overview
        overview = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'data_source': 'Dataroma (no external APIs)',
            'total_managers': holdings_df['manager'].nunique(),
            'total_unique_stocks': holdings_df['ticker'].nunique(),
            'total_positions': len(holdings_df),
            'positions_with_portfolio_pct': (holdings_df['percent_num'] > 0).sum(),
            'positions_with_price_data': (holdings_df['current_price'] > 0).sum(),
            'positions_with_market_cap': (holdings_df['market_cap'] > 0).sum(),
            'positions_with_sector': (holdings_df['sector'] != 'Unknown').sum(),
            'positions_with_gain_loss': (holdings_df['gain_loss_pct'] != 0).sum(),
            'market_cap_distribution': holdings_df['market_cap_category'].value_counts().to_dict(),
            'sector_distribution': holdings_df['sector'].value_counts().head(10).to_dict(),
            'top_10_most_held': conviction_df.head(10)[
                ['company', 'manager_count', 'avg_portfolio_pct', 'conviction_score']
            ].to_dict('index'),
            'top_5_by_conviction': conviction_df.head(5)[
                ['company', 'conviction_score', 'manager_count', 'avg_portfolio_pct']
            ].to_dict('index'),
            'top_5_concentrated': high_concentration.head(5)[
                ['manager', 'company', 'portfolio_pct']
            ].to_dict('index') if not high_concentration.empty else {},
            'top_5_hidden_gems': hidden_gems.head(5)[
                ['company', 'managers', 'avg_portfolio_pct']
            ].to_dict('index') if not hidden_gems.empty else {},
            'top_5_gainers': reports.get('top_gainers', pd.DataFrame()).head(5)[
                ['company', 'avg_gain_pct', 'manager_count']
            ].to_dict('index') if 'top_gainers' in reports else {},
            'top_5_losers': reports.get('top_losers', pd.DataFrame()).head(5)[
                ['company', 'avg_loss_pct', 'manager_count']
            ].to_dict('index') if 'top_losers' in reports else {}
        }
        reports['overview'] = overview
        
        return reports
    
    def save_all_reports(self, reports):
        """Save all analysis reports"""
        print("\nSaving analysis reports...")
        
        # Create master folder
        master_folder = os.path.join(self.analysis_dir, 'master_stock_analysis')
        
        # Save overview
        if 'overview' in reports:
            with open(os.path.join(self.analysis_dir, 'overview.json'), 'w') as f:
                json.dump(reports['overview'], f, indent=2, default=str)
            with open(os.path.join(master_folder, 'overview.json'), 'w') as f:
                json.dump(reports['overview'], f, indent=2, default=str)
        
        # Save each report as CSV
        for report_name, report_data in reports.items():
            if report_name == 'overview':
                continue
                
            if isinstance(report_data, pd.DataFrame):
                # Convert list columns to strings for CSV
                df = report_data.copy()
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else x
                        )
                
                # Save in both locations
                filename = os.path.join(self.analysis_dir, f'{report_name}.csv')
                df.to_csv(filename)
                
                master_filename = os.path.join(master_folder, f'{report_name}.csv')
                df.to_csv(master_filename)
                
                if not df.empty:
                    print(f"  Saved {report_name}.csv ({len(df)} rows)")
                else:
                    print(f"  Saved {report_name}.csv (empty)")
        
        # Create Excel file with all reports
        excel_file = os.path.join(self.analysis_dir, 'dataroma_analysis.xlsx')
        master_excel_file = os.path.join(master_folder, 'dataroma_complete_analysis.xlsx')
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Write overview
            overview_df = pd.DataFrame([reports.get('overview', {})])
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Write other reports
            sheet_order = [
                'high_conviction_stocks',
                'multi_manager_favorites',
                'top_gainers',
                'top_losers',
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
                    if not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name[:31])
        
        # Copy to master folder
        shutil.copy2(excel_file, master_excel_file)
        
        # Create README
        readme_content = """# Dataroma Stock Analysis Reports

This folder contains individual CSV files extracted from the complete Dataroma analysis.
All data is sourced directly from Dataroma.com without using external APIs.

## Files Generated:

### Overview
- `overview.json` - Summary statistics and key findings

### Price Threshold Analysis
- `stocks_under_$5.csv` through `stocks_under_$300.csv` - Stocks by price range

### Market Cap Analysis
- `micro-cap_favorites.csv` through `mega-cap_favorites.csv` - Stocks by market cap

### Conviction Analysis
- `high_conviction_stocks.csv` - Stocks with highest manager conviction
- `hidden_gems.csv` - Under-the-radar high-conviction picks
- `multi_manager_favorites.csv` - Stocks held by 5+ managers
- `highest_portfolio_concentration.csv` - Largest individual positions
- `managers_with_concentrated_bets.csv` - Managers with concentrated positions

### Performance Analysis
- `top_gainers.csv` - Best performers vs purchase price
- `top_losers.csv` - Worst performers vs purchase price

### Other Analysis
- `sector_analysis.csv` - Breakdown by sector

### Complete Analysis
- `dataroma_complete_analysis.xlsx` - All reports in one Excel file

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".format(datetime=datetime)
        
        readme_file = os.path.join(master_folder, 'README.md')
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"\nMaster Excel file saved: {excel_file}")
        print(f"Individual CSV files saved in: {master_folder}")
        
        # Summary
        generated_count = sum(1 for r in reports.values() if isinstance(r, pd.DataFrame) and not r.empty)
        empty_count = sum(1 for r in reports.values() if isinstance(r, pd.DataFrame) and r.empty)
        print(f"\nReports summary:")
        print(f"- Generated with data: {generated_count}")
        print(f"- Generated empty: {empty_count}")
        print(f"- Total files: {generated_count + empty_count + 1}")  # +1 for overview.json

def main():
    parser = argparse.ArgumentParser(description='Analyze Dataroma holdings')
    parser.add_argument('--mode', choices=['current', 'incremental', 'full'], 
                       default='current', 
                       help='Analysis mode')
    args = parser.parse_args()
    
    analyzer = DataromaAnalyzer()
    
    print("=" * 50)
    print("DATAROMA HOLDINGS ANALYZER")
    print("=" * 50)
    print("Using pure Dataroma data (no external APIs)")
    
    # For now, all modes do the same thing - analyze current holdings
    current_reports = analyzer.analyze_current_holdings()
    
    # Save all reports
    analyzer.save_all_reports(current_reports)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)
    
    overview = current_reports.get('overview', {})
    print(f"\nKey findings:")
    print(f"- Total managers: {overview.get('total_managers', 0)}")
    print(f"- Total stocks: {overview.get('total_unique_stocks', 0)}")
    print(f"- Total positions: {overview.get('total_positions', 0)}")
    
    print(f"\nData completeness:")
    print(f"- With portfolio %: {overview.get('positions_with_portfolio_pct', 0)}")
    print(f"- With prices: {overview.get('positions_with_price_data', 0)}")
    print(f"- With market cap: {overview.get('positions_with_market_cap', 0)}")
    print(f"- With sector: {overview.get('positions_with_sector', 0)}")
    
    if 'high_conviction_stocks' in current_reports and not current_reports['high_conviction_stocks'].empty:
        top_conviction = current_reports['high_conviction_stocks'].head(5)
        print(f"\nTop 5 High Conviction Stocks:")
        for ticker, row in top_conviction.iterrows():
            print(f"  {ticker}: {row['company']} - {row['manager_count']} managers, "
                  f"{row['avg_portfolio_pct']:.1f}% avg position, score: {row['conviction_score']:.1f}")

if __name__ == "__main__":
    main()