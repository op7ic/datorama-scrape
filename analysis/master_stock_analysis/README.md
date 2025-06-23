# Dataroma Stock Analysis Reports

Generated on: 2025-06-23 16:35:04

## Summary Statistics

- **Total Managers Tracked**: 80
- **Total Unique Stocks**: 1,508
- **Total Positions**: 3,223
- **Average Position Size**: 2.90%

## Data Quality

- Positions with Portfolio %: 3,215
- Positions with Price Data: 0
- Positions with Market Cap: 0
- Positions with Sector Data: 3,223

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
