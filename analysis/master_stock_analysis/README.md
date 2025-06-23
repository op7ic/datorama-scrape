# Dataroma Stock Analysis Reports

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
