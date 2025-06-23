import pandas as pd
import os
from datetime import datetime
import json
import argparse
import glob
import numpy as np
from collections import defaultdict
import shutil
import warnings
warnings.filterwarnings('ignore')

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
        if pd.isna(market_cap) or market_cap <= 0:
            return 'Unknown'
        elif market_cap < 300_000_000:
            return 'Micro-Cap (<$300M)'
        elif market_cap < 2_000_000_000:
            return 'Small-Cap ($300M-$2B)'
        elif market_cap < 10_000_000_000:
            return 'Mid-Cap ($2B-$10B)'
        elif market_cap < 200_000_000_000:
            return 'Large-Cap ($10B-$200B)'
        else:
            return 'Mega-Cap (>$200B)'
    
    def format_market_cap(self, market_cap):
        """Format market cap for display"""
        if pd.isna(market_cap) or market_cap <= 0:
            return 'N/A'
        elif market_cap >= 1_000_000_000_000:
            return f'${market_cap/1_000_000_000_000:.2f}T'
        elif market_cap >= 1_000_000_000:
            return f'${market_cap/1_000_000_000:.2f}B'
        elif market_cap >= 1_000_000:
            return f'${market_cap/1_000_000:.2f}M'
        else:
            return f'${market_cap:,.0f}'
    
    def get_latest_files(self, n=2):
        """Get the n most recent holdings files"""
        files = glob.glob(os.path.join(self.holdings_dir, 'holdings_*.csv'))
        files.sort(reverse=True)
        return files[:n]
    
    def load_and_prepare_data(self, filepath):
        """Load and prepare holdings data with proper type conversion"""
        print(f"\nLoading data from: {os.path.basename(filepath)}")
        
        holdings_df = pd.read_csv(filepath)
        print(f"Loaded {len(holdings_df)} raw holdings")
        
        # Convert numeric columns safely
        numeric_conversions = {
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
        
        for orig_col, new_col in numeric_conversions.items():
            if orig_col in holdings_df.columns:
                if orig_col == 'shares':
                    # Handle shares with commas and text
                    holdings_df[new_col] = holdings_df[orig_col].apply(
                        lambda x: pd.to_numeric(
                            str(x).replace(',', '').replace(' ', '').strip() 
                            if pd.notna(x) else '0', 
                            errors='coerce'
                        )
                    ).fillna(0)
                else:
                    holdings_df[new_col] = pd.to_numeric(
                        holdings_df[orig_col], errors='coerce'
                    ).fillna(0)
            else:
                holdings_df[new_col] = 0
        
        # Ensure required columns exist
        required_columns = {
            'ticker': '',
            'stock': '',
            'company': '',
            'manager': '',
            'sector': 'Unknown',
            'industry': 'Unknown',
            'recent_activity': '',
            'portfolio_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        for col, default_value in required_columns.items():
            if col not in holdings_df.columns:
                holdings_df[col] = default_value
        
        # Create company column if it doesn't exist
        if 'company' not in holdings_df.columns or holdings_df['company'].isna().all():
            holdings_df['company'] = holdings_df['stock']
        
        # Clean up ticker column
        holdings_df['ticker'] = holdings_df['ticker'].str.upper().str.strip()
        
        # Add calculated columns
        holdings_df['market_cap_category'] = holdings_df['market_cap'].apply(self.categorize_market_cap)
        holdings_df['market_cap_display'] = holdings_df['market_cap'].apply(self.format_market_cap)
        
        # Calculate gain/loss
        holdings_df['gain_loss_pct'] = 0
        mask = (holdings_df['reported_price'] > 0) & (holdings_df['current_price'] > 0)
        if mask.any():
            holdings_df.loc[mask, 'gain_loss_pct'] = (
                (holdings_df.loc[mask, 'current_price'] - holdings_df.loc[mask, 'reported_price']) / 
                holdings_df.loc[mask, 'reported_price'] * 100
            ).round(2)
        
        # Calculate position values
        holdings_df['position_value'] = holdings_df['shares_num'] * holdings_df['current_price']
        
        # Add value at risk (simplified)
        holdings_df['value_at_risk'] = holdings_df['position_value'] * 0.1  # 10% VaR assumption
        
        # Remove any completely empty rows
        holdings_df = holdings_df[holdings_df['ticker'].notna() & (holdings_df['ticker'] != '')]
        
        print(f"\nData Quality Summary:")
        print(f"- Valid holdings: {len(holdings_df)}")
        print(f"- Holdings with portfolio %: {(holdings_df['percent_num'] > 0).sum()}")
        print(f"- Holdings with current price: {(holdings_df['current_price'] > 0).sum()}")
        print(f"- Holdings with market cap: {(holdings_df['market_cap'] > 0).sum()}")
        print(f"- Holdings with sector: {(holdings_df['sector'] != 'Unknown').sum()}")
        print(f"- Holdings with gain/loss: {(holdings_df['gain_loss_pct'] != 0).sum()}")
        
        return holdings_df
    
    def analyze_current_holdings(self):
        """Analyze current holdings using only Dataroma data"""
        print("\n" + "="*60)
        print("ANALYZING CURRENT HOLDINGS")
        print("="*60)
        
        # Get latest holdings file
        latest_files = self.get_latest_files(1)
        if not latest_files:
            print("ERROR: No holdings files found in 'holdings' directory")
            return {}
        
        # Load and prepare data
        holdings_df = self.load_and_prepare_data(latest_files[0])
        
        if holdings_df.empty:
            print("ERROR: No valid holdings data found")
            return {}
        
        # Generate all reports
        return self.generate_all_reports(holdings_df)
    
    def generate_all_reports(self, holdings_df):
        """Generate all analysis reports"""
        print("\nGenerating analysis reports...")
        reports = {}
        
        # 1. Price Threshold Analysis
        print("\n1. Generating price threshold analysis...")
        price_thresholds = [5, 10, 20, 50, 100, 200, 300]
        for threshold in price_thresholds:
            stocks_under = holdings_df[
                (holdings_df['current_price'] > 0) & 
                (holdings_df['current_price'] <= threshold)
            ].copy()
            
            if not stocks_under.empty:
                summary = stocks_under.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())[:10]],  # Limit to 10 managers
                    'company': 'first',
                    'current_price': 'first',
                    'percent_num': ['mean', 'max'],
                    'shares_num': 'sum',
                    'market_cap': 'first',
                    'market_cap_category': 'first',
                    'sector': 'first',
                    'gain_loss_pct': 'mean',
                    'position_value': 'sum'
                }).round(2)
                
                summary.columns = ['manager_count', 'managers', 'company', 'price', 
                                 'avg_portfolio_pct', 'max_portfolio_pct', 'total_shares',
                                 'market_cap', 'cap_category', 'sector', 'avg_gain_loss_pct',
                                 'total_position_value']
                
                # Add formatted market cap
                summary['market_cap_display'] = summary['market_cap'].apply(self.format_market_cap)
                
                # Sort by manager count then by average portfolio percentage
                summary = summary.sort_values(['manager_count', 'avg_portfolio_pct'], ascending=[False, False])
                
                reports[f'stocks_under_${threshold}'] = summary
                print(f"  - Found {len(summary)} stocks under ${threshold}")
            else:
                reports[f'stocks_under_${threshold}'] = pd.DataFrame()
                print(f"  - No stocks found under ${threshold}")
        
        # 2. Conviction Score Analysis
        print("\n2. Generating conviction analysis...")
        conviction_df = holdings_df.groupby('ticker').agg({
            'manager': ['count', lambda x: list(x.unique())[:20]],  # Limit manager list
            'company': 'first',
            'current_price': 'first',
            'percent_num': ['mean', 'max', 'std'],
            'shares_num': 'sum',
            'market_cap': 'first',
            'market_cap_category': 'first',
            'sector': 'first',
            'industry': 'first',
            'position_value': 'sum'
        }).round(2)
        
        conviction_df.columns = ['manager_count', 'managers', 'company', 'price', 
                                'avg_portfolio_pct', 'max_portfolio_pct', 'portfolio_pct_std',
                                'total_shares', 'market_cap', 'cap_category', 'sector', 
                                'industry', 'total_position_value']
        
        # Calculate conviction score (weighted formula)
        conviction_df['conviction_score'] = (
            conviction_df['manager_count'] * 3 +  # Weight manager count more
            conviction_df['avg_portfolio_pct'] * 2 +  # Weight average position
            conviction_df['max_portfolio_pct'] * 0.5  # Consider max position
        ).round(2)
        
        # Add consistency score (lower std is better)
        conviction_df['portfolio_pct_std'] = conviction_df['portfolio_pct_std'].fillna(0)
        conviction_df['consistency_score'] = np.where(
            conviction_df['portfolio_pct_std'] > 0,
            conviction_df['avg_portfolio_pct'] / (1 + conviction_df['portfolio_pct_std']),
            conviction_df['avg_portfolio_pct']
        ).round(2)
        
        # Add formatted market cap
        conviction_df['market_cap_display'] = conviction_df['market_cap'].apply(self.format_market_cap)
        
        conviction_df = conviction_df.sort_values('conviction_score', ascending=False)
        reports['high_conviction_stocks'] = conviction_df.head(100)
        print(f"  - Generated conviction scores for {len(conviction_df)} stocks")
        
        # 3. Market Cap Categories
        print("\n3. Generating market cap analysis...")
        cap_categories = ['Micro-Cap (<$300M)', 'Small-Cap ($300M-$2B)', 
                         'Mid-Cap ($2B-$10B)', 'Large-Cap ($10B-$200B)', 
                         'Mega-Cap (>$200B)']
        
        for cap_category in cap_categories:
            cap_stocks = holdings_df[
                holdings_df['market_cap_category'] == cap_category
            ]
            
            if not cap_stocks.empty:
                cap_summary = cap_stocks.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())[:10]],
                    'company': 'first',
                    'current_price': 'first',
                    'percent_num': ['mean', 'max'],
                    'market_cap': 'first',
                    'sector': 'first',
                    'position_value': 'sum'
                }).round(2)
                
                cap_summary.columns = ['manager_count', 'managers', 'company', 'price',
                                     'avg_portfolio_pct', 'max_portfolio_pct', 'market_cap', 
                                     'sector', 'total_position_value']
                
                # Add formatted market cap
                cap_summary['market_cap_display'] = cap_summary['market_cap'].apply(self.format_market_cap)
                
                cap_summary = cap_summary.sort_values('manager_count', ascending=False)
                
                # Create filename-safe version of category name
                safe_category = cap_category.split(' ')[0].lower()
                reports[f'{safe_category}_favorites'] = cap_summary.head(50)
                print(f"  - Found {len(cap_summary)} {cap_category} stocks")
            else:
                safe_category = cap_category.split(' ')[0].lower()
                reports[f'{safe_category}_favorites'] = pd.DataFrame()
        
        # 4. Highest Portfolio Concentration
        print("\n4. Generating concentration analysis...")
        high_concentration = holdings_df[holdings_df['percent_num'] >= 5].copy()
        
        if not high_concentration.empty:
            high_concentration = high_concentration.sort_values('percent_num', ascending=False)
            concentration_report = high_concentration[
                ['manager', 'ticker', 'company', 'percent_num', 'current_price', 
                 'shares_num', 'position_value', 'market_cap_category', 'sector', 'gain_loss_pct']
            ].copy()
            concentration_report.columns = ['manager', 'ticker', 'company', 'portfolio_pct', 
                                          'price', 'shares', 'position_value', 'cap_category', 
                                          'sector', 'gain_loss_pct']
            reports['highest_portfolio_concentration'] = concentration_report.head(100)
            print(f"  - Found {len(concentration_report)} concentrated positions (≥5%)")
        else:
            # Fall back to top positions regardless of percentage
            top_positions = holdings_df[holdings_df['percent_num'] > 0].nlargest(100, 'percent_num')
            if not top_positions.empty:
                reports['highest_portfolio_concentration'] = top_positions[
                    ['manager', 'ticker', 'company', 'percent_num', 'current_price', 
                     'market_cap_category', 'sector', 'gain_loss_pct']
                ]
                print(f"  - Found {len(top_positions)} top positions")
            else:
                reports['highest_portfolio_concentration'] = pd.DataFrame()
        
        # 5. Hidden Gems
        print("\n5. Finding hidden gems...")
        # Hidden gems: Low manager count but high average position size
        hidden_gems_candidates = conviction_df[
            (conviction_df['manager_count'] <= 3) & 
            (conviction_df['avg_portfolio_pct'] >= 3)
        ]
        
        if hidden_gems_candidates.empty:
            # Alternative criteria
            hidden_gems_candidates = conviction_df[
                (conviction_df['manager_count'] <= 2) & 
                (conviction_df['avg_portfolio_pct'] >= 1)
            ]
        
        if not hidden_gems_candidates.empty:
            hidden_gems = hidden_gems_candidates.sort_values('avg_portfolio_pct', ascending=False).head(50)
            reports['hidden_gems'] = hidden_gems
            print(f"  - Found {len(hidden_gems)} hidden gems")
        else:
            reports['hidden_gems'] = pd.DataFrame()
            print("  - No hidden gems found with current criteria")
        
        # 6. Sector Analysis
        print("\n6. Generating sector analysis...")
        sector_summary = holdings_df[holdings_df['sector'] != 'Unknown'].groupby('sector').agg({
            'ticker': 'nunique',
            'manager': ['count', 'nunique'],
            'percent_num': 'mean',
            'position_value': 'sum',
            'market_cap': 'mean'
        }).round(2)
        
        if not sector_summary.empty:
            sector_summary.columns = ['unique_stocks', 'total_positions', 'unique_managers',
                                     'avg_portfolio_pct', 'total_value', 'avg_market_cap']
            sector_summary['avg_market_cap_display'] = sector_summary['avg_market_cap'].apply(self.format_market_cap)
            sector_summary = sector_summary.sort_values('total_positions', ascending=False)
            reports['sector_analysis'] = sector_summary
            print(f"  - Analyzed {len(sector_summary)} sectors")
        else:
            reports['sector_analysis'] = pd.DataFrame()
        
        # 7. Multi-Manager Holdings
        print("\n7. Finding multi-manager favorites...")
        multi_manager = conviction_df[conviction_df['manager_count'] >= 5].copy()
        if multi_manager.empty:
            multi_manager = conviction_df[conviction_df['manager_count'] >= 3].copy()
        
        if not multi_manager.empty:
            multi_manager = multi_manager.sort_values('manager_count', ascending=False).head(50)
            reports['multi_manager_favorites'] = multi_manager
            print(f"  - Found {len(multi_manager)} multi-manager stocks")
        else:
            reports['multi_manager_favorites'] = pd.DataFrame()
        
        # 8. Concentrated Positions by Manager
        print("\n8. Analyzing manager concentration...")
        concentrated = holdings_df[holdings_df['percent_num'] >= 5].copy()
        
        if not concentrated.empty:
            manager_concentration = concentrated.groupby('manager').agg({
                'ticker': ['count', lambda x: list(x)[:10]],  # Limit ticker list
                'percent_num': ['sum', 'mean', 'max'],
                'position_value': 'sum'
            }).round(2)
            
            manager_concentration.columns = ['concentrated_positions', 'top_tickers', 
                                           'total_portfolio_pct', 'avg_position_size',
                                           'largest_position', 'total_value']
            manager_concentration = manager_concentration.sort_values('concentrated_positions', ascending=False)
            reports['managers_with_concentrated_bets'] = manager_concentration.head(50)
            print(f"  - Found {len(manager_concentration)} managers with concentrated bets")
        else:
            reports['managers_with_concentrated_bets'] = pd.DataFrame()
        
        # 9. Gainers and Losers
        print("\n9. Analyzing performance...")
        if (holdings_df['gain_loss_pct'] != 0).any():
            # Top Gainers
            gainers = holdings_df[holdings_df['gain_loss_pct'] > 5].copy()  # At least 5% gain
            if not gainers.empty:
                top_gainers = gainers.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())[:10]],
                    'company': 'first',
                    'reported_price': 'mean',
                    'current_price': 'first',
                    'gain_loss_pct': 'mean',
                    'percent_num': 'mean',
                    'market_cap': 'first',
                    'sector': 'first'
                }).round(2)
                
                top_gainers.columns = ['manager_count', 'managers', 'company', 
                                      'avg_reported_price', 'current_price', 
                                      'avg_gain_pct', 'avg_portfolio_pct',
                                      'market_cap', 'sector']
                top_gainers['market_cap_display'] = top_gainers['market_cap'].apply(self.format_market_cap)
                top_gainers = top_gainers.sort_values('avg_gain_pct', ascending=False)
                reports['top_gainers'] = top_gainers.head(50)
                print(f"  - Found {len(top_gainers)} gaining stocks")
            else:
                reports['top_gainers'] = pd.DataFrame()
            
            # Top Losers
            losers = holdings_df[holdings_df['gain_loss_pct'] < -5].copy()  # At least 5% loss
            if not losers.empty:
                top_losers = losers.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())[:10]],
                    'company': 'first',
                    'reported_price': 'mean',
                    'current_price': 'first',
                    'gain_loss_pct': 'mean',
                    'percent_num': 'mean',
                    'market_cap': 'first',
                    'sector': 'first'
                }).round(2)
                
                top_losers.columns = ['manager_count', 'managers', 'company', 
                                     'avg_reported_price', 'current_price', 
                                     'avg_loss_pct', 'avg_portfolio_pct',
                                     'market_cap', 'sector']
                top_losers['market_cap_display'] = top_losers['market_cap'].apply(self.format_market_cap)
                top_losers = top_losers.sort_values('avg_loss_pct', ascending=True)
                reports['top_losers'] = top_losers.head(50)
                print(f"  - Found {len(top_losers)} losing stocks")
            else:
                reports['top_losers'] = pd.DataFrame()
        else:
            reports['top_gainers'] = pd.DataFrame()
            reports['top_losers'] = pd.DataFrame()
            print("  - No gain/loss data available")
        
        # 10. Overview
        print("\n10. Generating overview...")
        overview = self.generate_overview(holdings_df, reports, conviction_df)
        reports['overview'] = overview
        
        print("\nReport generation complete!")
        return reports
    
    def generate_overview(self, holdings_df, reports, conviction_df):
        """Generate comprehensive overview statistics"""
        overview = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'Dataroma.com (no external APIs)',
            'total_managers': int(holdings_df['manager'].nunique()),
            'total_unique_stocks': int(holdings_df['ticker'].nunique()),
            'total_positions': len(holdings_df),
            'positions_with_portfolio_pct': int((holdings_df['percent_num'] > 0).sum()),
            'positions_with_price_data': int((holdings_df['current_price'] > 0).sum()),
            'positions_with_market_cap': int((holdings_df['market_cap'] > 0).sum()),
            'positions_with_sector': int((holdings_df['sector'] != 'Unknown').sum()),
            'positions_with_gain_loss': int((holdings_df['gain_loss_pct'] != 0).sum()),
            'total_position_value': float(holdings_df['position_value'].sum()),
            'average_position_size': float(holdings_df[holdings_df['percent_num'] > 0]['percent_num'].mean()) 
                if (holdings_df['percent_num'] > 0).any() else 0,
            'market_cap_distribution': holdings_df['market_cap_category'].value_counts().to_dict(),
            'sector_distribution': holdings_df[holdings_df['sector'] != 'Unknown']['sector'].value_counts().head(10).to_dict()
        }
        
        # Add top stocks summaries
        if not conviction_df.empty:
            top10 = conviction_df.head(10)
            overview['top_10_most_held'] = {}
            for ticker, row in top10.iterrows():
                overview['top_10_most_held'][ticker] = {
                    'company': row['company'],
                    'manager_count': int(row['manager_count']),
                    'avg_portfolio_pct': float(row['avg_portfolio_pct']),
                    'conviction_score': float(row['conviction_score']),
                    'sector': row['sector'],
                    'market_cap': self.format_market_cap(row['market_cap'])
                }
        
        # Add other top lists
        self._add_top_list(overview, 'top_5_by_conviction', conviction_df.head(5), 
                          ['company', 'conviction_score', 'manager_count', 'avg_portfolio_pct'])
        
        if 'highest_portfolio_concentration' in reports and not reports['highest_portfolio_concentration'].empty:
            conc_df = reports['highest_portfolio_concentration'].head(5)
            overview['top_5_concentrated'] = {}
            for idx, row in conc_df.iterrows():
                key = f"{row['manager']}_{row['ticker']}"
                overview['top_5_concentrated'][key] = {
                    'manager': row['manager'],
                    'company': row['company'],
                    'ticker': row['ticker'],
                    'portfolio_pct': float(row['portfolio_pct'])
                }
        
        if 'hidden_gems' in reports and not reports['hidden_gems'].empty:
            self._add_top_list(overview, 'top_5_hidden_gems', reports['hidden_gems'].head(5),
                              ['company', 'managers', 'avg_portfolio_pct'])
        
        if 'top_gainers' in reports and not reports['top_gainers'].empty:
            self._add_top_list(overview, 'top_5_gainers', reports['top_gainers'].head(5),
                              ['company', 'avg_gain_pct', 'manager_count'])
        
        if 'top_losers' in reports and not reports['top_losers'].empty:
            self._add_top_list(overview, 'top_5_losers', reports['top_losers'].head(5),
                              ['company', 'avg_loss_pct', 'manager_count'])
        
        return overview
    
    def _add_top_list(self, overview, key, df, columns):
        """Helper to add top lists to overview"""
        if df.empty:
            overview[key] = {}
            return
            
        overview[key] = {}
        for ticker, row in df.iterrows():
            item_data = {}
            for col in columns:
                if col in row:
                    value = row[col]
                    if isinstance(value, list):
                        value = ', '.join(value[:5])  # Limit lists
                    elif pd.api.types.is_numeric_dtype(type(value)):
                        value = float(value)
                    else:
                        value = str(value)
                    item_data[col] = value
            overview[key][ticker] = item_data
    
    def save_all_reports(self, reports):
        """Save all analysis reports"""
        print("\n" + "="*60)
        print("SAVING ANALYSIS REPORTS")
        print("="*60)
        
        # Create master folder
        master_folder = os.path.join(self.analysis_dir, 'master_stock_analysis')
        
        # Save overview
        if 'overview' in reports:
            overview_path = os.path.join(self.analysis_dir, 'overview.json')
            with open(overview_path, 'w') as f:
                json.dump(reports['overview'], f, indent=2, default=str)
            
            master_overview_path = os.path.join(master_folder, 'overview.json')
            with open(master_overview_path, 'w') as f:
                json.dump(reports['overview'], f, indent=2, default=str)
            print("✓ Saved overview.json")
        
        # Save each report as CSV
        csv_count = 0
        for report_name, report_data in reports.items():
            if report_name == 'overview':
                continue
                
            if isinstance(report_data, pd.DataFrame) and not report_data.empty:
                # Convert list columns to strings for CSV
                df = report_data.copy()
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(
                            lambda x: ', '.join(map(str, x[:10])) if isinstance(x, list) else x
                        )
                
                # Save in both locations
                filename = os.path.join(self.analysis_dir, f'{report_name}.csv')
                df.to_csv(filename)
                
                master_filename = os.path.join(master_folder, f'{report_name}.csv')
                df.to_csv(master_filename)
                
                csv_count += 1
                print(f"✓ Saved {report_name}.csv ({len(df)} rows)")
        
        # Create comprehensive Excel file
        print("\nCreating Excel reports...")
        excel_file = os.path.join(self.analysis_dir, 'dataroma_analysis.xlsx')
        master_excel_file = os.path.join(master_folder, 'dataroma_complete_analysis.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Write overview as formatted DataFrame
                overview_data = []
                for key, value in reports.get('overview', {}).items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            overview_data.append({
                                'Category': key,
                                'Item': sub_key,
                                'Value': str(sub_value)
                            })
                    else:
                        overview_data.append({
                            'Category': 'General',
                            'Item': key,
                            'Value': str(value)
                        })
                
                if overview_data:
                    overview_df = pd.DataFrame(overview_data)
                    overview_df.to_excel(writer, sheet_name='Overview', index=False)
                
                # Define sheet order
                sheet_order = [
                    'high_conviction_stocks',
                    'multi_manager_favorites',
                    'top_gainers',
                    'top_losers',
                    'hidden_gems',
                    'stocks_under_$10',
                    'stocks_under_$20',
                    'stocks_under_$50',
                    'stocks_under_$100',
                    'micro-cap_favorites',
                    'small-cap_favorites',
                    'mid-cap_favorites',
                    'large-cap_favorites',
                    'mega-cap_favorites',
                    'highest_portfolio_concentration',
                    'managers_with_concentrated_bets',
                    'sector_analysis'
                ]
                
                # Write data sheets
                sheets_written = 0
                for sheet_name in sheet_order:
                    if sheet_name in reports and isinstance(reports[sheet_name], pd.DataFrame):
                        df = reports[sheet_name].copy()
                        if not df.empty:
                            # Convert lists to strings
                            for col in df.columns:
                                if df[col].dtype == 'object':
                                    df[col] = df[col].apply(
                                        lambda x: ', '.join(map(str, x[:10])) if isinstance(x, list) else str(x)
                                    )
                            
                            # Truncate sheet name to Excel's 31 character limit
                            excel_sheet_name = sheet_name[:31]
                            df.to_excel(writer, sheet_name=excel_sheet_name)
                            sheets_written += 1
                
                print(f"✓ Created Excel file with {sheets_written + 1} sheets")
        
        except Exception as e:
            print(f"⚠ Error creating Excel file: {e}")
        
        # Copy to master folder
        if os.path.exists(excel_file):
            shutil.copy2(excel_file, master_excel_file)
            print(f"✓ Copied Excel to master folder")
        
        # Create comprehensive README
        self.create_readme(master_folder, reports)
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print(f"="*60)
        print(f"\nFiles saved to:")
        print(f"- Analysis directory: {self.analysis_dir}")
        print(f"- Master directory: {master_folder}")
        print(f"\nGenerated:")
        print(f"- {csv_count} CSV files")
        print(f"- 1 Excel workbook")
        print(f"- 1 JSON overview")
        print(f"- 1 README file")
    
    def create_readme(self, folder_path, reports):
        """Create comprehensive README file"""
        overview = reports.get('overview', {})
        
        readme_content = f"""# Dataroma Stock Analysis Reports

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- **Total Managers Tracked**: {overview.get('total_managers', 0):,}
- **Total Unique Stocks**: {overview.get('total_unique_stocks', 0):,}
- **Total Positions**: {overview.get('total_positions', 0):,}
- **Average Position Size**: {overview.get('average_position_size', 0):.2f}%

## Data Quality

- Positions with Portfolio %: {overview.get('positions_with_portfolio_pct', 0):,}
- Positions with Price Data: {overview.get('positions_with_price_data', 0):,}
- Positions with Market Cap: {overview.get('positions_with_market_cap', 0):,}
- Positions with Sector Data: {overview.get('positions_with_sector', 0):,}

## Files Generated

### Overview
- `overview.json` - Summary statistics and key findings

### Price Threshold Analysis
- `stocks_under_$5.csv` - Stocks priced under $5
- `stocks_under_$10.csv` - Stocks priced under $10
- `stocks_under_$20.csv` - Stocks priced under $20
- `stocks_under_$50.csv` - Stocks priced under $50
- `stocks_under_$100.csv` - Stocks priced under $100
- `stocks_under_$200.csv` - Stocks priced under $200
- `stocks_under_$300.csv` - Stocks priced under $300

### Market Cap Analysis
- `micro-cap_favorites.csv` - Micro-cap stocks (<$300M)
- `small-cap_favorites.csv` - Small-cap stocks ($300M-$2B)
- `mid-cap_favorites.csv` - Mid-cap stocks ($2B-$10B)
- `large-cap_favorites.csv` - Large-cap stocks ($10B-$200B)
- `mega-cap_favorites.csv` - Mega-cap stocks (>$200B)

### Conviction Analysis
- `high_conviction_stocks.csv` - Stocks with highest manager conviction scores
- `hidden_gems.csv` - Under-the-radar high-conviction picks (few managers, high %)
- `multi_manager_favorites.csv` - Stocks held by 5+ managers
- `highest_portfolio_concentration.csv` - Largest individual portfolio positions
- `managers_with_concentrated_bets.csv` - Managers with 5%+ concentrated positions

### Performance Analysis
- `top_gainers.csv` - Best performers vs reported purchase price
- `top_losers.csv` - Worst performers vs reported purchase price

### Other Analysis
- `sector_analysis.csv` - Breakdown by sector

### Complete Analysis
- `dataroma_complete_analysis.xlsx` - All reports in one Excel file with multiple sheets

## Data Source

All data sourced directly from Dataroma.com without using external APIs.
Analysis includes only the holdings data available from superinvestor portfolios.

## Notes

- Market cap categories follow standard definitions
- Conviction scores are calculated using manager count, average position size, and maximum position
- Hidden gems are stocks with few managers but high average portfolio allocation
- Performance metrics compare current prices to reported purchase prices where available
"""
        
        readme_file = os.path.join(folder_path, 'README.md')
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        print("✓ Created README.md")

def main():
    parser = argparse.ArgumentParser(description='Analyze Dataroma holdings')
    parser.add_argument('--mode', choices=['current', 'incremental', 'full'], 
                       default='current', 
                       help='Analysis mode (currently all modes analyze latest data)')
    args = parser.parse_args()
    
    analyzer = DataromaAnalyzer()
    
    print("=" * 60)
    print("DATAROMA HOLDINGS ANALYZER")
    print("=" * 60)
    print("Using pure Dataroma data (no external APIs)")
    print(f"Mode: {args.mode}")
    
    # Analyze current holdings
    current_reports = analyzer.analyze_current_holdings()
    
    if current_reports:
        # Save all reports
        analyzer.save_all_reports(current_reports)
        
        # Print summary
        overview = current_reports.get('overview', {})
        print(f"\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        print(f"\nPortfolio Overview:")
        print(f"- Total managers: {overview.get('total_managers', 0)}")
        print(f"- Total stocks: {overview.get('total_unique_stocks', 0)}")
        print(f"- Total positions: {overview.get('total_positions', 0)}")
        print(f"- Average position size: {overview.get('average_position_size', 0):.2f}%")
        
        if 'top_10_most_held' in overview and overview['top_10_most_held']:
            print(f"\nTop 5 Most Widely Held Stocks:")
            for i, (ticker, data) in enumerate(list(overview['top_10_most_held'].items())[:5], 1):
                print(f"{i}. {ticker}: {data['company']} - {data['manager_count']} managers, "
                      f"{data['avg_portfolio_pct']:.1f}% avg position")
        
        if 'top_5_gainers' in overview and overview['top_5_gainers']:
            print(f"\nTop Gainers:")
            for ticker, data in list(overview['top_5_gainers'].items())[:3]:
                print(f"  {ticker}: +{data.get('avg_gain_pct', 0):.1f}% "
                      f"({data.get('manager_count', 0)} managers)")
    else:
        print("\nERROR: No reports generated. Check if holdings data exists.")

if __name__ == "__main__":
    main()