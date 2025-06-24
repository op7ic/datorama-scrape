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
    
    def categorize_pe_ratio(self, pe_ratio):
        """Categorize stocks by P/E ratio"""
        if pd.isna(pe_ratio) or pe_ratio <= 0:
            return 'N/A or Negative'
        elif pe_ratio < 10:
            return 'Value (<10)'
        elif pe_ratio < 20:
            return 'Fair Value (10-20)'
        elif pe_ratio < 35:
            return 'Growth (20-35)'
        else:
            return 'High Growth (>35)'
    
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
        
        # Ensure all expected columns exist with defaults
        column_defaults = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_date': datetime.now().strftime('%Y-%m-%d'),
            'manager': '',
            'ticker': '',
            'company': '',
            'stock': '',
            'portfolio_percent': 0.0,
            'shares': 0,
            'recent_activity': '',
            'reported_price': 0.0,
            'current_price': 0.0,
            'value': 0.0,
            'company_name': '',
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0.0,
            'pe_ratio': 0.0,
            'ownership_count': 0
        }
        
        for col, default_value in column_defaults.items():
            if col not in holdings_df.columns:
                holdings_df[col] = default_value
        
        # Clean up ticker column
        holdings_df['ticker'] = holdings_df['ticker'].astype(str).str.upper().str.strip()
        
        # Convert numeric columns safely
        numeric_columns = ['portfolio_percent', 'shares', 'reported_price', 'current_price', 
                          'value', 'market_cap', 'pe_ratio', 'ownership_count']
        
        for col in numeric_columns:
            holdings_df[col] = pd.to_numeric(holdings_df[col], errors='coerce').fillna(0)
        
        # Use company_name if company is empty
        mask = (holdings_df['company'].isna()) | (holdings_df['company'] == '')
        holdings_df.loc[mask, 'company'] = holdings_df.loc[mask, 'company_name']
        
        # If both are empty, use stock column
        mask = (holdings_df['company'].isna()) | (holdings_df['company'] == '')
        holdings_df.loc[mask, 'company'] = holdings_df.loc[mask, 'stock']
        
        # Add calculated columns
        holdings_df['market_cap_category'] = holdings_df['market_cap'].apply(self.categorize_market_cap)
        holdings_df['market_cap_display'] = holdings_df['market_cap'].apply(self.format_market_cap)
        holdings_df['pe_category'] = holdings_df['pe_ratio'].apply(self.categorize_pe_ratio)
        
        # Calculate gain/loss
        holdings_df['gain_loss_pct'] = 0.0
        mask = (holdings_df['reported_price'] > 0) & (holdings_df['current_price'] > 0)
        if mask.any():
            holdings_df.loc[mask, 'gain_loss_pct'] = (
                (holdings_df.loc[mask, 'current_price'] - holdings_df.loc[mask, 'reported_price']) / 
                holdings_df.loc[mask, 'reported_price'] * 100
            ).round(2)
        
        # Calculate position values
        holdings_df['position_value'] = holdings_df['shares'] * holdings_df['current_price']
        
        # Calculate value metrics
        holdings_df['price_to_book'] = 0.0  # Would need book value data
        holdings_df['peg_ratio'] = 0.0  # Would need growth rate data
        
        # Activity flags
        holdings_df['is_new'] = holdings_df['recent_activity'].str.contains('New', case=False, na=False)
        holdings_df['is_add'] = holdings_df['recent_activity'].str.contains('Add|Buy', case=False, na=False)
        holdings_df['is_reduce'] = holdings_df['recent_activity'].str.contains('Reduce|Sell|Trim', case=False, na=False)
        holdings_df['is_exit'] = holdings_df['recent_activity'].str.contains('Exit|Sold All|Sold Out', case=False, na=False)
        
        # Remove any completely empty rows
        holdings_df = holdings_df[holdings_df['ticker'].notna() & (holdings_df['ticker'] != '')]
        
        print(f"\nData Quality Summary:")
        print(f"- Valid holdings: {len(holdings_df)}")
        print(f"- Holdings with portfolio %: {(holdings_df['portfolio_percent'] > 0).sum()}")
        print(f"- Holdings with current price: {(holdings_df['current_price'] > 0).sum()}")
        print(f"- Holdings with market cap: {(holdings_df['market_cap'] > 0).sum()}")
        print(f"- Holdings with sector: {(holdings_df['sector'] != 'Unknown').sum()}")
        print(f"- Holdings with P/E ratio: {(holdings_df['pe_ratio'] > 0).sum()}")
        print(f"- Holdings with ownership count: {(holdings_df['ownership_count'] > 0).sum()}")
        print(f"- Holdings with gain/loss: {(holdings_df['gain_loss_pct'] != 0).sum()}")
        print(f"- New positions: {holdings_df['is_new'].sum()}")
        print(f"- Added positions: {holdings_df['is_add'].sum()}")
        print(f"- Reduced positions: {holdings_df['is_reduce'].sum()}")
        
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
                    'manager': ['count', lambda x: list(x.unique())[:10]],
                    'company': 'first',
                    'current_price': 'first',
                    'portfolio_percent': ['mean', 'max'],
                    'shares': 'sum',
                    'market_cap': 'first',
                    'market_cap_category': 'first',
                    'sector': 'first',
                    'industry': 'first',
                    'pe_ratio': 'first',
                    'ownership_count': 'first',
                    'gain_loss_pct': 'mean',
                    'position_value': 'sum',
                    'recent_activity': lambda x: ', '.join(x[x != ''].unique()[:5])
                }).round(2)
                
                summary.columns = ['manager_count', 'managers', 'company', 'price', 
                                 'avg_portfolio_pct', 'max_portfolio_pct', 'total_shares',
                                 'market_cap', 'cap_category', 'sector', 'industry',
                                 'pe_ratio', 'ownership_count', 'avg_gain_loss_pct',
                                 'total_position_value', 'recent_activities']
                
                summary['market_cap_display'] = summary['market_cap'].apply(self.format_market_cap)
                summary = summary.sort_values(['manager_count', 'avg_portfolio_pct'], ascending=[False, False])
                
                reports[f'stocks_under_${threshold}'] = summary
                print(f"  - Found {len(summary)} stocks under ${threshold}")
        
        # 2. Enhanced Conviction Score Analysis
        print("\n2. Generating enhanced conviction analysis...")
        conviction_df = holdings_df.groupby('ticker').agg({
            'manager': ['count', lambda x: list(x.unique())[:20]],
            'company': 'first',
            'current_price': 'first',
            'portfolio_percent': ['mean', 'max', 'std'],
            'shares': 'sum',
            'market_cap': 'first',
            'market_cap_category': 'first',
            'sector': 'first',
            'industry': 'first',
            'pe_ratio': 'first',
            'ownership_count': 'first',
            'position_value': 'sum',
            'is_new': 'sum',
            'is_add': 'sum',
            'is_reduce': 'sum'
        }).round(2)
        
        conviction_df.columns = ['manager_count', 'managers', 'company', 'price', 
                                'avg_portfolio_pct', 'max_portfolio_pct', 'portfolio_pct_std',
                                'total_shares', 'market_cap', 'cap_category', 'sector', 
                                'industry', 'pe_ratio', 'ownership_count', 'total_position_value',
                                'new_positions', 'added_positions', 'reduced_positions']
        
        # Enhanced conviction score with ownership count
        conviction_df['conviction_score'] = (
            conviction_df['manager_count'] * 3 +
            conviction_df['avg_portfolio_pct'] * 2 +
            conviction_df['max_portfolio_pct'] * 0.5 +
            (conviction_df['ownership_count'] / 10)  # Bonus for high ownership
        ).round(2)
        
        # Momentum score based on activity
        conviction_df['momentum_score'] = (
            conviction_df['new_positions'] * 2 +
            conviction_df['added_positions'] -
            conviction_df['reduced_positions'] * 2
        ).round(2)
        
        conviction_df['market_cap_display'] = conviction_df['market_cap'].apply(self.format_market_cap)
        conviction_df = conviction_df.sort_values('conviction_score', ascending=False)
        
        reports['high_conviction_stocks'] = conviction_df.head(100)
        print(f"  - Generated conviction scores for {len(conviction_df)} stocks")
        
        # 3. Market Cap Categories
        print("\n3. Generating market cap analysis...")
        for cap_category in ['Micro-Cap (<$300M)', 'Small-Cap ($300M-$2B)', 
                           'Mid-Cap ($2B-$10B)', 'Large-Cap ($10B-$200B)', 'Mega-Cap (>$200B)']:
            cap_stocks = holdings_df[holdings_df['market_cap_category'] == cap_category]
            
            if not cap_stocks.empty:
                cap_summary = cap_stocks.groupby('ticker').agg({
                    'manager': ['count', lambda x: list(x.unique())[:10]],
                    'company': 'first',
                    'current_price': 'first',
                    'portfolio_percent': ['mean', 'max'],
                    'market_cap': 'first',
                    'sector': 'first',
                    'pe_ratio': 'first',
                    'ownership_count': 'first',
                    'position_value': 'sum'
                }).round(2)
                
                cap_summary.columns = ['manager_count', 'managers', 'company', 'price',
                                     'avg_portfolio_pct', 'max_portfolio_pct', 'market_cap', 
                                     'sector', 'pe_ratio', 'ownership_count', 'total_position_value']
                
                cap_summary['market_cap_display'] = cap_summary['market_cap'].apply(self.format_market_cap)
                cap_summary = cap_summary.sort_values('manager_count', ascending=False)
                
                safe_category = cap_category.split(' ')[0].lower()
                reports[f'{safe_category}_favorites'] = cap_summary.head(50)
                print(f"  - Found {len(cap_summary)} {cap_category} stocks")
        
        # 4. Highest Portfolio Concentration
        print("\n4. Generating concentration analysis...")
        high_concentration = holdings_df[holdings_df['portfolio_percent'] >= 5].copy()
        
        if not high_concentration.empty:
            concentration_report = high_concentration.sort_values('portfolio_percent', ascending=False)
            concentration_report = concentration_report[[
                'manager', 'ticker', 'company', 'portfolio_percent', 'current_price', 
                'shares', 'position_value', 'market_cap_category', 'sector', 
                'pe_ratio', 'ownership_count', 'gain_loss_pct', 'recent_activity'
            ]].head(100)
            
            reports['highest_portfolio_concentration'] = concentration_report
            print(f"  - Found {len(high_concentration)} concentrated positions (≥5%)")
        
        # 5. Hidden Gems (Enhanced)
        print("\n5. Finding enhanced hidden gems...")
        # Hidden gems: Low manager count but high conviction and good metrics
        hidden_gems_candidates = conviction_df[
            (conviction_df['manager_count'] <= 3) & 
            (conviction_df['avg_portfolio_pct'] >= 3) &
            (conviction_df['ownership_count'] >= 5)  # At least 5 total owners
        ]
        
        if not hidden_gems_candidates.empty:
            hidden_gems = hidden_gems_candidates.sort_values('conviction_score', ascending=False).head(50)
            reports['hidden_gems'] = hidden_gems
            print(f"  - Found {len(hidden_gems)} hidden gems")
        
        # 6. Sector and Industry Analysis
        print("\n6. Generating sector analysis...")
        sector_summary = holdings_df[holdings_df['sector'] != 'Unknown'].groupby('sector').agg({
            'ticker': 'nunique',
            'manager': ['count', 'nunique'],
            'portfolio_percent': 'mean',
            'position_value': 'sum',
            'market_cap': 'mean',
            'pe_ratio': 'mean',
            'gain_loss_pct': 'mean'
        }).round(2)
        
        if not sector_summary.empty:
            sector_summary.columns = ['unique_stocks', 'total_positions', 'unique_managers',
                                     'avg_portfolio_pct', 'total_value', 'avg_market_cap',
                                     'avg_pe_ratio', 'avg_gain_loss']
            sector_summary['avg_market_cap_display'] = sector_summary['avg_market_cap'].apply(self.format_market_cap)
            sector_summary = sector_summary.sort_values('total_positions', ascending=False)
            reports['sector_analysis'] = sector_summary
            print(f"  - Analyzed {len(sector_summary)} sectors")
        
        # 7. Multi-Manager Holdings with Momentum
        print("\n7. Finding multi-manager favorites with momentum...")
        multi_manager = conviction_df[conviction_df['manager_count'] >= 5].copy()
        
        if not multi_manager.empty:
            # Sort by combination of manager count and momentum
            multi_manager['combined_score'] = (
                multi_manager['manager_count'] * 0.6 + 
                multi_manager['momentum_score'] * 0.4
            )
            multi_manager = multi_manager.sort_values('combined_score', ascending=False).head(50)
            reports['multi_manager_favorites'] = multi_manager
            print(f"  - Found {len(multi_manager)} multi-manager stocks")
        
        # 8. Value Opportunities (Low P/E with high conviction)
        print("\n8. Finding value opportunities...")
        value_stocks = conviction_df[
            (conviction_df['pe_ratio'] > 0) & 
            (conviction_df['pe_ratio'] < 15) &
            (conviction_df['manager_count'] >= 2)
        ].copy()
        
        if not value_stocks.empty:
            value_stocks = value_stocks.sort_values('conviction_score', ascending=False).head(50)
            reports['value_opportunities'] = value_stocks
            print(f"  - Found {len(value_stocks)} value opportunities")
        
        # 9. Recent Activity Analysis
        print("\n9. Analyzing recent activity...")
        # New positions
        new_positions = holdings_df[holdings_df['is_new']].copy()
        if not new_positions.empty:
            new_summary = new_positions.groupby('ticker').agg({
                'manager': ['count', lambda x: list(x.unique())[:10]],
                'company': 'first',
                'portfolio_percent': 'mean',
                'current_price': 'first',
                'market_cap_category': 'first',
                'sector': 'first'
            })
            new_summary.columns = ['manager_count', 'managers', 'company', 
                                  'avg_portfolio_pct', 'price', 'cap_category', 'sector']
            new_summary = new_summary.sort_values('manager_count', ascending=False).head(50)
            reports['new_positions'] = new_summary
            print(f"  - Found {len(new_summary)} stocks with new positions")
        
        # 10. Ownership Analysis
        print("\n10. Analyzing ownership patterns...")
        high_ownership = holdings_df[holdings_df['ownership_count'] >= 10].copy()
        
        if not high_ownership.empty:
            ownership_summary = high_ownership.groupby('ticker').agg({
                'ownership_count': 'first',
                'manager': 'count',
                'company': 'first',
                'portfolio_percent': 'mean',
                'market_cap': 'first',
                'sector': 'first',
                'pe_ratio': 'first'
            }).round(2)
            
            ownership_summary.columns = ['total_owners', 'dataroma_managers', 'company',
                                        'avg_portfolio_pct', 'market_cap', 'sector', 'pe_ratio']
            ownership_summary['market_cap_display'] = ownership_summary['market_cap'].apply(self.format_market_cap)
            ownership_summary['ownership_ratio'] = (ownership_summary['dataroma_managers'] / ownership_summary['total_owners'] * 100).round(1)
            ownership_summary = ownership_summary.sort_values('total_owners', ascending=False).head(50)
            reports['high_ownership_stocks'] = ownership_summary
            print(f"  - Found {len(ownership_summary)} high ownership stocks")
        
        # 11. Generate Overview
        print("\n11. Generating comprehensive overview...")
        overview = self.generate_overview(holdings_df, reports, conviction_df)
        reports['overview'] = overview
        
        print("\nReport generation complete!")
        return reports
    
    def generate_overview(self, holdings_df, reports, conviction_df):
        """Generate comprehensive overview statistics"""
        overview = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'Dataroma.com (enhanced with stock metadata)',
            'total_managers': int(holdings_df['manager'].nunique()),
            'total_unique_stocks': int(holdings_df['ticker'].nunique()),
            'total_positions': len(holdings_df),
            'positions_with_portfolio_pct': int((holdings_df['portfolio_percent'] > 0).sum()),
            'positions_with_price_data': int((holdings_df['current_price'] > 0).sum()),
            'positions_with_market_cap': int((holdings_df['market_cap'] > 0).sum()),
            'positions_with_sector': int((holdings_df['sector'] != 'Unknown').sum()),
            'positions_with_pe_ratio': int((holdings_df['pe_ratio'] > 0).sum()),
            'positions_with_ownership_data': int((holdings_df['ownership_count'] > 0).sum()),
            'positions_with_gain_loss': int((holdings_df['gain_loss_pct'] != 0).sum()),
            'new_positions': int(holdings_df['is_new'].sum()),
            'added_positions': int(holdings_df['is_add'].sum()),
            'reduced_positions': int(holdings_df['is_reduce'].sum()),
            'total_position_value': float(holdings_df['position_value'].sum()),
            'average_position_size': float(holdings_df[holdings_df['portfolio_percent'] > 0]['portfolio_percent'].mean()) 
                if (holdings_df['portfolio_percent'] > 0).any() else 0,
            'average_pe_ratio': float(holdings_df[holdings_df['pe_ratio'] > 0]['pe_ratio'].mean())
                if (holdings_df['pe_ratio'] > 0).any() else 0,
            'market_cap_distribution': holdings_df['market_cap_category'].value_counts().to_dict(),
            'sector_distribution': holdings_df[holdings_df['sector'] != 'Unknown']['sector'].value_counts().head(10).to_dict(),
            'pe_distribution': holdings_df['pe_category'].value_counts().to_dict()
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
                    'momentum_score': float(row.get('momentum_score', 0)),
                    'sector': row['sector'],
                    'pe_ratio': float(row.get('pe_ratio', 0)),
                    'ownership_count': int(row.get('ownership_count', 0)),
                    'market_cap': self.format_market_cap(row['market_cap'])
                }
        
        # Add activity summary
        activity_summary = {
            'most_added': [],
            'most_reduced': [],
            'most_new': []
        }
        
        if 'new_positions' in reports and not reports['new_positions'].empty:
            for ticker, row in reports['new_positions'].head(5).iterrows():
                activity_summary['most_new'].append({
                    'ticker': ticker,
                    'company': row['company'],
                    'managers': row['manager_count']
                })
        
        overview['recent_activity'] = activity_summary
        
        return overview
    
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
                    'value_opportunities',
                    'new_positions',
                    'high_ownership_stocks',
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
- **Average P/E Ratio**: {overview.get('average_pe_ratio', 0):.1f}

## Data Quality

- Positions with Portfolio %: {overview.get('positions_with_portfolio_pct', 0):,}
- Positions with Price Data: {overview.get('positions_with_price_data', 0):,}
- Positions with Market Cap: {overview.get('positions_with_market_cap', 0):,}
- Positions with Sector Data: {overview.get('positions_with_sector', 0):,}
- Positions with P/E Ratio: {overview.get('positions_with_pe_ratio', 0):,}
- Positions with Ownership Data: {overview.get('positions_with_ownership_data', 0):,}

## Recent Activity

- New Positions: {overview.get('new_positions', 0):,}
- Added Positions: {overview.get('added_positions', 0):,}
- Reduced Positions: {overview.get('reduced_positions', 0):,}

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

### Conviction & Momentum Analysis
- `high_conviction_stocks.csv` - Stocks with highest manager conviction scores
- `hidden_gems.csv` - Under-the-radar high-conviction picks (few managers, high %)
- `multi_manager_favorites.csv` - Stocks held by 5+ managers with momentum scores
- `highest_portfolio_concentration.csv` - Largest individual portfolio positions
- `value_opportunities.csv` - Low P/E stocks with high conviction

### Activity Analysis
- `new_positions.csv` - Stocks with new positions from managers
- `high_ownership_stocks.csv` - Stocks with high total ownership count

### Other Analysis
- `sector_analysis.csv` - Breakdown by sector with performance metrics

### Complete Analysis
- `dataroma_complete_analysis.xlsx` - All reports in one Excel file with multiple sheets

## Data Source

All data sourced directly from Dataroma.com including:
- Portfolio holdings and percentages
- Share counts and prices
- Market capitalization
- P/E ratios and valuation metrics
- Sector and industry classifications
- Total ownership counts
- Recent activity (new, add, reduce, exit)

## Scoring Methodology

### Conviction Score
Calculated using:
- Manager count (weight: 3x)
- Average portfolio percentage (weight: 2x)
- Maximum portfolio percentage (weight: 0.5x)
- Total ownership count (weight: 0.1x)

### Momentum Score
Based on recent activity:
- New positions (+2 points each)
- Added positions (+1 point each)
- Reduced positions (-2 points each)

## Notes

- Market cap categories follow standard definitions
- P/E ratios are used to identify value opportunities
- Hidden gems are stocks with few managers but high average portfolio allocation
- Activity flags help identify trending positions
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
    print("DATAROMA HOLDINGS ANALYZER (Enhanced)")
    print("=" * 60)
    print("Using Dataroma data with enriched metadata")
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
        print(f"- Average P/E ratio: {overview.get('average_pe_ratio', 0):.1f}")
        
        print(f"\nRecent Activity:")
        print(f"- New positions: {overview.get('new_positions', 0)}")
        print(f"- Added positions: {overview.get('added_positions', 0)}")
        print(f"- Reduced positions: {overview.get('reduced_positions', 0)}")
        
        if 'top_10_most_held' in overview and overview['top_10_most_held']:
            print(f"\nTop 5 Most Widely Held Stocks:")
            for i, (ticker, data) in enumerate(list(overview['top_10_most_held'].items())[:5], 1):
                print(f"{i}. {ticker}: {data['company']} - {data['manager_count']} managers, "
                      f"{data['avg_portfolio_pct']:.1f}% avg position, "
                      f"P/E: {data['pe_ratio']:.1f}")
    else:
        print("\nERROR: No reports generated. Check if holdings data exists.")

if __name__ == "__main__":
    main()