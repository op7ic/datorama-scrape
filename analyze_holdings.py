#!/usr/bin/env python3
"""
Portfolio holdings analysis module.

This module analyzes investment portfolio data from Dataroma,
generating comprehensive reports, visualizations, and insights.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# Progress Update:
# ✓ Set up imports with proper ordering and type hints
# → Working on manager name mapping and core functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Manager name mapping for human-readable output
MANAGER_NAMES = {
    "abc": "Abrams Capital Management",
    "AC": "Akre Capital Management",
    "AIM": "Atlantic Investment Management",
    "AKO": "AKO Capital",
    "AM": "Appaloosa Management",
    "AP": "AltaRock Partners",
    "aq": "Aquamarine Capital",
    "ARFFX": "Ariel Focus Fund",
    "BAUPOST": "Baupost Group",
    "BRK": "Berkshire Hathaway",
    "ca": "Chou Associates",
    "CAAPX": "Ariel Appreciation Fund",
    "CAS": "CAS Investment Partners",
    "CAU": "Causeway Capital Management",
    "cc": "Cantillon Capital Management",
    "CCM": "Brave Warrior Advisors",
    "CM": "Conifer Management",
    "DA": "Dorsey Asset Management",
    "DAV": "Davis Advisors",
    "DODGX": "Dodge & Cox",
    "EC": "Egerton Capital",
    "ENG": "Engaged Capital",
    "fairx": "Fairholme Capital",
    "FE": "First Eagle Investment Management",
    "FFH": "Fairfax Financial Holdings",
    "FPACX": "FPA Crescent Fund",
    "FPPTX": "FPA Queens Road Small Cap Value Fund",
    "FS": "Fundsmith",
    "GA": "Greenhaven Associates",
    "GC": "Giverny Capital",
    "GFT": "Bill & Melinda Gates Foundation Trust",
    "GLC": "Greenlea Lane Capital",
    "GLRE": "Greenlight Capital",
    "GR": "Gardner Russo & Quinn",
    "HC": "Himalaya Capital Management",
    "hcmax": "Hillman Value Fund",
    "HH": "H&H International Investment",
    "ic": "Icahn Capital Management",
    "JIM": "Jensen Investment Management",
    "KB": "Kahn Brothers Group",
    "LLPFX": "Longleaf Partners",
    "LMM": "Miller Value Partners",
    "LPC": "Lone Pine Capital",
    "LT": "Lindsell Train",
    "MAVFX": "Matrix Asset Advisors",
    "mc": "Maverick Capital",
    "MKL": "Markel Group",
    "MP": "Makaira Partners",
    "MPGFX": "Mairs & Power Growth Fund",
    "MVALX": "Meridian Contrarian Fund",
    "oa": "Leon Cooperman",
    "oaklx": "Oakmark Select Fund",
    "oc": "Oaktree Capital Management",
    "OCL": "Oakcliff Capital",
    "OFALX": "Olstein Capital Management",
    "PC": "Punch Card Management",
    "pcm": "Polen Capital Management",
    "PI": "Pabrai Investments",
    "psc": "Pershing Square Capital Management",
    "PTNT": "Patient Capital Management",
    "pzfvx": "Hancock Classic Value",
    "RVC": "RV Capital GmbH",
    "SA": "Semper Augustus",
    "SAM": "Scion Asset Management",
    "SEQUX": "Sequoia Fund",
    "SP": "ShawSpring Partners",
    "SSHFX": "Sound Shore",
    "T": "Torray Funds",
    "TA": "Third Avenue Management",
    "tci": "TCI Fund Management",
    "TF": "Trian Fund Management",
    "TFP": "Triple Frond Partners",
    "TGM": "Tiger Global Management",
    "tp": "Third Point",
    "TWEBX": "Tweedy Browne Value Fund",
    "VA": "ValueAct Capital",
    "VFC": "Valley Forge Capital Management",
    "vg": "Viking Global Investors",
    "WP": "Wedgewood Partners",
    "WVALX": "Weitz Large Cap Equity Fund",
    "YAM": "Yacktman Asset Management",
}


class HoldingsAnalyzer:
    """Analyzes portfolio holdings data and generates insights."""

    def __init__(self, cache_dir: str = "cache", output_dir: str = "analysis") -> None:
        """
        Initialize the holdings analyzer.

        Args:
            cache_dir: Directory containing cached data files
            output_dir: Directory for output files and visualizations
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.visual_dir = self.output_dir / "visuals"
        self.historical_dir = self.output_dir / "historical"
        self.historical_visual_dir = self.historical_dir / "visuals"

        # Ensure output directories exist
        self._ensure_directories()

        # Load data
        self.holdings_df: Optional[pd.DataFrame] = None
        self.history_df: Optional[pd.DataFrame] = None
        self.stocks_df: Optional[pd.DataFrame] = None
        self.managers_df: Optional[pd.DataFrame] = None
        self.manager_name_map: dict[str, str] = {}

        # Configure visualization style
        # plt.style.use("seaborn-v0_8-darkgrid")  # Not needed for current analysis
        # sns.set_palette("husl")  # Not needed for current analysis
    
    @property
    def holdings(self) -> pd.DataFrame:
        """Get holdings dataframe."""
        return self.holdings_df if self.holdings_df is not None else pd.DataFrame()

    def _ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        directories = [
            self.output_dir,
            self.visual_dir,
            self.historical_dir,
            self.historical_visual_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load all required data from cache files."""
        logging.info("Loading data from cache files...")

        # Load holdings data
        holdings_file = self.cache_dir / "holdings.json"
        if holdings_file.exists():
            with open(holdings_file) as f:
                holdings_data = json.load(f)
                self.holdings_df = pd.DataFrame(holdings_data)
                
                # Convert numeric columns and add defaults for missing ones
                if not self.holdings_df.empty:
                    numeric_columns = ["shares", "value", "portfolio_percent"]
                    for col in numeric_columns:
                        if col in self.holdings_df.columns:
                            self.holdings_df[col] = pd.to_numeric(
                                self.holdings_df[col], errors="coerce"
                            ).fillna(0)
                        else:
                            # Add default column if missing
                            self.holdings_df[col] = 0
                
                logging.info(f"Loaded {len(self.holdings_df)} holdings")
        else:
            raise FileNotFoundError(f"Holdings file not found: {holdings_file}")

        # Load history data
        history_file = self.cache_dir / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                history_data = json.load(f)
                self.history_df = pd.DataFrame(history_data)
                logging.info(f"Loaded {len(self.history_df)} historical activities")

        # Load stocks data
        stocks_file = self.cache_dir / "stocks.json"
        if stocks_file.exists():
            with open(stocks_file) as f:
                stocks_data = json.load(f)
                # Convert dict to DataFrame
                stocks_list = []
                for ticker, data in stocks_data.items():
                    data["ticker"] = ticker
                    stocks_list.append(data)
                self.stocks_df = pd.DataFrame(stocks_list)
                logging.info(f"Loaded {len(self.stocks_df)} stocks")

        # Load managers data
        managers_file = self.cache_dir / "managers.json"
        if managers_file.exists():
            with open(managers_file) as f:
                managers_data = json.load(f)
                # Handle both list and dict formats
                if isinstance(managers_data, list):
                    self.managers_df = pd.DataFrame(managers_data)
                else:
                    # Convert dict to DataFrame
                    managers_list = []
                    for manager_id, data in managers_data.items():
                        data["manager_id"] = manager_id
                        managers_list.append(data)
                    self.managers_df = pd.DataFrame(managers_list)
                logging.info(f"Loaded {len(self.managers_df)} managers")

        # Add manager names to dataframes
        self._add_manager_names()

    def _add_manager_names(self) -> None:
        """Add human-readable manager names to dataframes."""
        # Create manager name mapping - prioritize hardcoded names
        self.manager_name_map = MANAGER_NAMES.copy()
        
        # Update with any additional managers from JSON that have real names
        if self.managers_df is not None:
            for _, row in self.managers_df.iterrows():
                manager_id = row.get("id", row.get("manager_id", ""))
                name = row.get("name", "")
                # Only use JSON name if it's different from the ID and not empty
                if name and name != manager_id and manager_id not in self.manager_name_map:
                    self.manager_name_map[manager_id] = name

        if self.holdings_df is not None and not self.holdings_df.empty:
            # Use 'manager' column if 'manager_id' doesn't exist
            if "manager_id" in self.holdings_df.columns:
                manager_col = "manager_id"
            elif "manager" in self.holdings_df.columns:
                manager_col = "manager"
            else:
                # No manager column found - add a default one
                self.holdings_df["manager_id"] = "unknown"
                manager_col = "manager_id"
            
            self.holdings_df["manager_name"] = self.holdings_df[manager_col].map(
                lambda x: self.manager_name_map.get(x, x)
            )
            # Rename to standard column name
            if manager_col == "manager":
                self.holdings_df["manager_id"] = self.holdings_df["manager"]

        if self.history_df is not None and not self.history_df.empty:
            # Use 'manager' column if 'manager_id' doesn't exist
            if "manager_id" in self.history_df.columns:
                manager_col = "manager_id"
            elif "manager" in self.history_df.columns:
                manager_col = "manager"
            else:
                # No manager column found - add a default one
                self.history_df["manager_id"] = "unknown"
                manager_col = "manager_id"
            
            self.history_df["manager_name"] = self.history_df[manager_col].map(
                lambda x: self.manager_name_map.get(x, x)
            )
            # Rename to standard column name
            if manager_col == "manager":
                self.history_df["manager_id"] = self.history_df["manager"]

    def get_manager_list(self, manager_ids: pd.Series) -> str:
        """
        Get formatted list of manager names from IDs.

        Args:
            manager_ids: Series of manager IDs

        Returns:
            Comma-separated string of manager names (limited to 10)
        """
        unique_managers = list(manager_ids.unique())
        manager_names = [self.manager_name_map.get(m, m) for m in unique_managers]
        return ", ".join(manager_names[:10])  # Limit display to 10

    def get_activity_summary(self, activities: pd.Series) -> str:
        """
        Get summary of recent activities.

        Args:
            activities: Series of activity strings

        Returns:
            Semicolon-separated string of unique activities
        """
        clean_activities = []
        for activity in activities.dropna().unique():
            activity_str = str(activity).strip()
            if (
                activity_str
                and activity_str != "nan"
                and not re.match(r"^[A-Z0-9.-]+\s*-\s*.+", activity_str)
            ):
                clean_activities.append(activity_str)
        return "; ".join(clean_activities[:5])

    def apply_precision_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply consistent numeric formatting to DataFrame columns.

        Args:
            df: DataFrame to format

        Returns:
            Formatted DataFrame
        """
        # Define precision rules
        precision_rules = {
            "price": 2,
            "avg_portfolio_pct": 2,
            "max_portfolio_pct": 2,
            "portfolio_pct_std": 2,
            "gain_loss_pct": 2,
            "pe_ratio": 2,
            "debt_to_equity": 2,
            "roe": 2,
            "current_ratio": 2,
            "gross_margin": 2,
            "beta": 2,
            "eps": 4,
            "conviction_score": 2,
            "value_score": 1,
            "market_cap": 0,
            "total_shares": 0,
            "total_position_value": 2,
            "manager_count": 0,
            "position_count": 0,
            "shares": 0,
            "num_managers": 0,
        }

        # Apply formatting
        for col, decimals in precision_rules.items():
            if col in df.columns:
                try:
                    # Convert to numeric first
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    # Round to specified decimals
                    df[col] = df[col].round(decimals)
                except Exception as e:
                    logging.warning(f"Could not format column {col}: {e}")

        return df

    def analyze_top_holdings(self) -> pd.DataFrame:
        """
        Analyze top holdings across all managers.

        Returns:
            DataFrame with top holdings analysis
        """
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # Check required columns
        required_columns = ["ticker", "manager_id", "shares", "value"]
        missing_columns = [col for col in required_columns if col not in self.holdings_df.columns]
        if missing_columns:
            logging.warning(f"Missing required columns for top holdings analysis: {missing_columns}")
            return pd.DataFrame()

        # Group by ticker
        grouped = self.holdings_df.groupby("ticker").agg(
            {
                "manager_id": ["count", self.get_manager_list],
                "shares": "sum",
                "value": "sum",
                "portfolio_percent": ["mean", "max", "std"],
                "portfolio_date": ["min", "max"],  # Add date range
            }
        )

        # Flatten column names
        grouped.columns = [
            "manager_count",
            "managers",
            "total_shares",
            "total_value",
            "avg_portfolio_pct",
            "max_portfolio_pct",
            "portfolio_pct_std",
            "earliest_report_date",
            "latest_report_date",
        ]

        # Add stock info if available
        if self.stocks_df is not None:
            # Use only available columns
            available_cols = [
                col
                for col in ["name", "market_cap", "pe_ratio", "sector", "industry"]
                if col in self.stocks_df.columns
            ]
            if available_cols:
                stock_info = self.stocks_df.set_index("ticker")[available_cols]
                grouped = grouped.join(stock_info, how="left")

        # Sort by manager count and value
        grouped = grouped.sort_values(
            ["manager_count", "total_value"], ascending=[False, False]
        )

        # Apply formatting
        grouped = self.apply_precision_formatting(grouped.reset_index())

        return grouped.head(50)

    def analyze_sector_distribution(self) -> pd.DataFrame:
        """
        Analyze sector distribution of holdings.

        Returns:
            DataFrame with sector analysis
        """
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")

        # Use holdings data directly if it has sector info
        if "sector" in self.holdings_df.columns:
            merged = self.holdings_df.copy()
        elif self.stocks_df is not None:
            # Merge holdings with stock info
            stock_cols = ["ticker", "sector"]
            if "industry" in self.stocks_df.columns:
                stock_cols.append("industry")
            merged = self.holdings_df.merge(
                self.stocks_df[stock_cols],
                on="ticker",
                how="left",
            )
        else:
            raise ValueError("No sector information available")

        # Group by sector
        sector_analysis = merged.groupby("sector").agg(
            {
                "value": "sum",
                "ticker": "nunique",
                "manager_id": "nunique",
            }
        )

        sector_analysis.columns = ["total_value", "unique_stocks", "unique_managers"]
        sector_analysis["avg_value_per_stock"] = (
            sector_analysis["total_value"] / sector_analysis["unique_stocks"]
        )

        # Sort by total value
        sector_analysis = sector_analysis.sort_values("total_value", ascending=False)

        return self.apply_precision_formatting(sector_analysis.reset_index())

    def analyze_manager_performance(self) -> pd.DataFrame:
        """
        Analyze manager performance metrics.

        Returns:
            DataFrame with manager performance analysis
        """
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")

        # Group by manager
        manager_stats = self.holdings_df.groupby("manager_id").agg(
            {
                "ticker": "count",
                "value": ["sum", "mean", "std"],
                "portfolio_percent": ["mean", "max"],
                "portfolio_date": "max",  # Latest reporting date
            }
        )

        # Flatten columns
        manager_stats.columns = [
            "position_count",
            "total_value",
            "avg_position_value",
            "value_std",
            "avg_portfolio_pct",
            "max_portfolio_pct",
            "latest_report_date",
        ]

        # Add manager names
        manager_stats["manager_name"] = manager_stats.index.map(
            lambda x: self.manager_name_map.get(x, x)
        )

        # Calculate concentration metrics
        manager_stats["concentration_score"] = (
            manager_stats["max_portfolio_pct"] / manager_stats["avg_portfolio_pct"]
        )
        
        # Filter out managers with 0.0 portfolio percentages (inactive managers)
        active_managers = manager_stats[
            (manager_stats["avg_portfolio_pct"] > 0.0) & 
            (manager_stats["max_portfolio_pct"] > 0.0)
        ].copy()

        # Sort by total value
        active_managers = active_managers.sort_values("total_value", ascending=False)

        return self.apply_precision_formatting(active_managers.reset_index())

    def analyze_historical_manager_performance(self) -> pd.DataFrame:
        """
        Analyze historical manager performance including inactive managers.

        Returns:
            DataFrame with all manager performance analysis including inactive ones
        """
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")

        # Group by manager (including inactive ones)
        manager_stats = self.holdings_df.groupby("manager_id").agg(
            {
                "ticker": "count",
                "value": ["sum", "mean", "std"],
                "portfolio_percent": ["mean", "max"],
                "portfolio_date": "max",  # Latest reporting date
            }
        )

        # Flatten columns
        manager_stats.columns = [
            "position_count",
            "total_value",
            "avg_position_value",
            "value_std",
            "avg_portfolio_pct",
            "max_portfolio_pct",
            "latest_report_date",
        ]

        # Add manager names
        manager_stats["manager_name"] = manager_stats.index.map(
            lambda x: self.manager_name_map.get(x, x)
        )

        # Calculate concentration metrics (handle division by zero for inactive managers)
        manager_stats["concentration_score"] = (
            manager_stats["max_portfolio_pct"] / manager_stats["avg_portfolio_pct"]
        ).fillna(0.0)
        
        # Filter for historical managers only (those with 0.0 portfolio percentages)
        historical_managers = manager_stats[
            (manager_stats["avg_portfolio_pct"] == 0.0) | 
            (manager_stats["max_portfolio_pct"] == 0.0)
        ].copy()

        # Sort by total value
        historical_managers = historical_managers.sort_values("total_value", ascending=False)

        return self.apply_precision_formatting(historical_managers.reset_index())

    def analyze_momentum_stocks(self) -> pd.DataFrame:
        """
        Identify stocks with momentum based on recent activity.

        Returns:
            DataFrame with momentum stock analysis
        """
        if self.history_df is None:
            raise ValueError("History data not loaded")

        # Filter recent activities (last 2 quarters)
        recent_activities = self.history_df[
            self.history_df["action"].str.contains("Add|New", case=False, na=False)
        ]

        # Group by ticker
        momentum = recent_activities.groupby("ticker").agg(
            {
                "manager_id": ["count", self.get_manager_list],
                "shares": "sum",
                "action": self.get_activity_summary,
                "period": lambda x: ", ".join(x.unique()),  # Show all periods
            }
        )

        momentum.columns = [
            "buy_count",
            "managers",
            "shares_added",
            "activities",
            "periods",
        ]

        # Add current holdings info
        if self.holdings_df is not None:
            current = self.holdings_df.groupby("ticker").agg(
                {
                    "value": "sum",
                    "portfolio_percent": "mean",
                    "recent_activity": lambda x: "; ".join(
                        [act for act in x.dropna() if act]
                    ),
                    "portfolio_date": "max",  # Most recent portfolio date
                }
            )
            current.columns = [
                "current_value",
                "avg_portfolio_pct",
                "recent_holdings_activity",
                "latest_portfolio_date",
            ]
            momentum = momentum.join(current, how="left")

        # Sort by buy count
        momentum = momentum.sort_values("buy_count", ascending=False)

        return self.apply_precision_formatting(momentum.reset_index()).head(30)

    def analyze_new_positions(self) -> pd.DataFrame:
        """
        Identify new positions with their initiation dates.

        Returns:
            DataFrame with new positions and when they were initiated
        """
        if self.history_df is None:
            raise ValueError("History data not loaded")

        # Filter for new positions (Buy actions indicate new positions)
        new_positions = self.history_df[self.history_df["action_type"] == "Buy"].copy()

        if len(new_positions) == 0:
            return pd.DataFrame()

        # Group by ticker and manager to get initiation details
        new_analysis = (
            new_positions.groupby(["ticker", "manager_id", "period"])
            .agg(
                {
                    "shares": "sum",
                    "value": "sum",
                    "action": "first",
                }
            )
            .reset_index()
        )

        # Add manager names
        new_analysis["manager_name"] = new_analysis["manager_id"].map(
            lambda x: self.manager_name_map.get(x, x)
        )

        # Sort by period (most recent first) and value
        new_analysis = new_analysis.sort_values(
            ["period", "value"], ascending=[False, False]
        )

        return self.apply_precision_formatting(new_analysis).head(50)

    def analyze_value_opportunities(self) -> pd.DataFrame:
        """
        Identify potential value opportunities.

        Returns:
            DataFrame with value opportunity analysis
        """
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")

        # Use holdings data directly if it has PE ratio
        if "pe_ratio" in self.holdings_df.columns:
            merged = self.holdings_df.copy()
        elif self.stocks_df is not None:
            # Merge holdings with stock metrics
            stock_cols = ["ticker"]
            for col in ["pe_ratio", "market_cap"]:
                if col in self.stocks_df.columns:
                    stock_cols.append(col)
            merged = self.holdings_df.merge(
                self.stocks_df[stock_cols],
                on="ticker",
                how="left",
            )
        else:
            return pd.DataFrame()  # Return empty if no PE data available

        # Filter for low P/E stocks held by multiple managers
        value_stocks = merged[
            (merged["pe_ratio"] > 0) & (merged["pe_ratio"] < 15)
        ].copy()

        # Group by ticker
        if len(value_stocks) > 0:
            agg_dict = {
                "manager_id": ["count", self.get_manager_list],
                "pe_ratio": "first",
                "value": "sum",
            }
            if "market_cap" in value_stocks.columns:
                agg_dict["market_cap"] = "first"
            if "stock" in value_stocks.columns:
                agg_dict["stock"] = "first"

            value_analysis = value_stocks.groupby("ticker").agg(agg_dict)

            # Flatten column names based on what was aggregated
            new_cols = ["manager_count", "managers", "pe_ratio"]
            if "market_cap" in agg_dict:
                new_cols.append("market_cap")
            if "stock" in agg_dict:
                new_cols.append("company_name")
            new_cols.append("total_value")
            value_analysis.columns = new_cols

            # Calculate value score
            value_analysis["value_score"] = (
                value_analysis["manager_count"] * 10 / value_analysis["pe_ratio"]
            )

            # Sort by value score
            value_analysis = value_analysis.sort_values("value_score", ascending=False)

            return self.apply_precision_formatting(value_analysis.reset_index()).head(
                30
            )
        return pd.DataFrame()
    
    def categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap into standard buckets."""
        if pd.isna(market_cap) or market_cap <= 0:
            return "Unknown"
        elif market_cap < 300_000_000:  # < $300M
            return "Micro-Cap"
        elif market_cap < 2_000_000_000:  # < $2B
            return "Small-Cap"
        elif market_cap < 10_000_000_000:  # < $10B
            return "Mid-Cap"
        elif market_cap < 200_000_000_000:  # < $200B
            return "Large-Cap"
        else:
            return "Mega-Cap"
    
    def format_market_cap(self, market_cap: float) -> str:
        """Format market cap for display."""
        if pd.isna(market_cap) or market_cap <= 0:
            return "N/A"
        elif market_cap >= 1_000_000_000_000:  # Trillion
            return f"${market_cap / 1_000_000_000_000:.1f}T"
        elif market_cap >= 1_000_000_000:  # Billion
            return f"${market_cap / 1_000_000_000:.1f}B"
        elif market_cap >= 1_000_000:  # Million
            return f"${market_cap / 1_000_000:.1f}M"
        else:
            return f"${market_cap:,.0f}"
    
    def analyze_high_conviction_stocks(self) -> pd.DataFrame:
        """Find stocks where managers have high conviction (large portfolio %)."""
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # Filter for positions > 5% of portfolio
        high_conviction = self.holdings_df[
            self.holdings_df["portfolio_percent"] > 5.0
        ].copy()
        
        if high_conviction.empty:
            return pd.DataFrame()
        
        # Add temporal context
        if self.history_df is not None:
            # Get first buy date for each position
            first_buys = self.history_df[
                self.history_df["action_type"] == "Buy"
            ].groupby(["ticker", "manager_id"])["period"].first().reset_index()
            first_buys.columns = ["ticker", "manager_id", "first_buy_period"]
            
            high_conviction = high_conviction.merge(
                first_buys, on=["ticker", "manager_id"], how="left"
            )
        
        # Group by ticker
        grouped = high_conviction.groupby("ticker").agg({
            "manager_id": ["count", self.get_manager_list],
            "portfolio_percent": ["mean", "max"],
            "value": "sum",
            "shares": "sum",
            "portfolio_date": "max",
        })
        
        # Flatten columns
        grouped.columns = [
            "manager_count", "managers", "avg_portfolio_pct", 
            "max_portfolio_pct", "total_value", "total_shares", "latest_date"
        ]
        
        # Calculate conviction score
        grouped["conviction_score"] = (
            grouped["avg_portfolio_pct"] * grouped["manager_count"]
        )
        
        # Add stock info if available
        if self.stocks_df is not None:
            grouped = grouped.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Sort by conviction score
        grouped = grouped.sort_values("conviction_score", ascending=False)
        
        # Add manager names and format
        if "managers" in grouped.columns:
            grouped["manager_names"] = grouped["managers"].apply(
                lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
            )
        
        return self.apply_precision_formatting(grouped.reset_index()).head(50)
    
    def analyze_interesting_stocks_overview(self) -> pd.DataFrame:
        """Create comprehensive overview of most interesting stocks."""
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # Start with all holdings
        overview = self.holdings_df.groupby("ticker").agg({
            "manager_id": ["count", self.get_manager_list],
            "shares": "sum",
            "value": "sum",
            "portfolio_percent": ["mean", "max"],
            "portfolio_date": "max",
        })
        
        overview.columns = [
            "manager_count", "managers", "total_shares", "total_value",
            "avg_portfolio_pct", "max_portfolio_pct", "latest_date"
        ]
        
        # Add stock info
        if self.stocks_df is not None:
            overview = overview.merge(
                self.stocks_df[["ticker", "market_cap", "pe_ratio", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Add recent activity if available
        if self.history_df is not None:
            recent_activity = self.history_df[
                self.history_df["action_type"].isin(["Buy", "Add"])
            ].groupby("ticker").agg({
                "period": ["count", "max"],
                "manager_id": "nunique"
            })
            recent_activity.columns = ["buy_count", "last_buy_period", "active_managers"]
            overview = overview.merge(recent_activity, left_index=True, right_index=True, how="left")
            overview["buy_count"] = overview["buy_count"].fillna(0)
        
        # Calculate appeal score (0-10)
        overview["appeal_score"] = 0
        
        # Manager count factor (max 3 points)
        overview["appeal_score"] += (overview["manager_count"] / 10).clip(0, 3)
        
        # Portfolio percentage factor (max 2 points)
        overview["appeal_score"] += (overview["avg_portfolio_pct"] / 5).clip(0, 2)
        
        # Value factor (max 2 points)
        if "pe_ratio" in overview.columns:
            value_score = overview["pe_ratio"].apply(
                lambda x: 2 if 0 < x < 15 else 1 if 15 <= x < 25 else 0
            )
            overview["appeal_score"] += value_score
        
        # Recent activity factor (max 3 points)
        if "buy_count" in overview.columns:
            overview["appeal_score"] += (overview["buy_count"] / 20).clip(0, 3)
        
        # Investment timing
        overview["investment_timing"] = "Consider"
        if "last_buy_period" in overview.columns:
            overview.loc[overview["last_buy_period"] == "Q4 2024", "investment_timing"] = "Good"
            overview.loc[overview["buy_count"] > 10, "investment_timing"] = "Strong"
        
        # Add manager names
        overview["manager_names"] = overview["managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
        )
        
        # Sort by appeal score
        overview = overview.sort_values("appeal_score", ascending=False)
        
        return self.apply_precision_formatting(overview.reset_index()).head(100)
    
    def analyze_hidden_gems(self) -> pd.DataFrame:
        """Find under-followed quality stocks."""
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # Find stocks with 1-3 managers
        holdings_by_ticker = self.holdings_df.groupby("ticker").agg({
            "manager_id": ["count", self.get_manager_list],
            "portfolio_percent": ["mean", "max"],
            "value": "sum",
            "shares": "sum",
        })
        
        holdings_by_ticker.columns = [
            "manager_count", "managers", "avg_portfolio_pct",
            "max_portfolio_pct", "total_value", "total_shares"
        ]
        
        # Filter for under-followed but high conviction
        hidden_gems = holdings_by_ticker[
            (holdings_by_ticker["manager_count"] <= 3) &
            (holdings_by_ticker["max_portfolio_pct"] > 3.0)
        ].copy()
        
        if hidden_gems.empty:
            return pd.DataFrame()
        
        # Add stock info
        if self.stocks_df is not None:
            hidden_gems = hidden_gems.merge(
                self.stocks_df[["ticker", "market_cap", "pe_ratio", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Categorize hidden type
        hidden_gems["hidden_type"] = "Under-followed"
        if "pe_ratio" in hidden_gems.columns:
            hidden_gems.loc[
                (hidden_gems["pe_ratio"] > 0) & (hidden_gems["pe_ratio"] < 15),
                "hidden_type"
            ] = "Value Play"
            hidden_gems.loc[
                (hidden_gems["pe_ratio"] > 0) & 
                (hidden_gems["pe_ratio"] < 15) &
                (hidden_gems["manager_count"] == 1),
                "hidden_type"
            ] = "Deep Value"
        
        # Add first buy date if available
        if self.history_df is not None:
            first_buys = self.history_df[
                self.history_df["action_type"] == "Buy"
            ].groupby("ticker")["period"].first()
            hidden_gems = hidden_gems.merge(
                first_buys.rename("first_seen_period"),
                left_index=True, right_index=True, how="left"
            )
        
        # Add manager names
        hidden_gems["manager_names"] = hidden_gems["managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
        )
        
        # Sort by portfolio percentage
        hidden_gems = hidden_gems.sort_values("max_portfolio_pct", ascending=False)
        
        return self.apply_precision_formatting(hidden_gems.reset_index()).head(50)
    
    def analyze_multi_manager_favorites(self) -> pd.DataFrame:
        """Find stocks held by 5+ managers (consensus picks)."""
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # Group by ticker
        grouped = self.holdings_df.groupby("ticker").agg({
            "manager_id": ["count", self.get_manager_list],
            "shares": "sum",
            "value": "sum",
            "portfolio_percent": ["mean", "max"],
            "portfolio_date": ["min", "max"],
        })
        
        grouped.columns = [
            "manager_count", "managers", "total_shares", "total_value",
            "avg_portfolio_pct", "max_portfolio_pct", "earliest_date", "latest_date"
        ]
        
        # Filter for 5+ managers
        multi_manager = grouped[grouped["manager_count"] >= 5].copy()
        
        if multi_manager.empty:
            return pd.DataFrame()
        
        # Add consensus score
        multi_manager["consensus_score"] = (
            multi_manager["manager_count"] * multi_manager["avg_portfolio_pct"]
        )
        
        # Add stock info
        if self.stocks_df is not None:
            multi_manager = multi_manager.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Add recent activity
        if self.history_df is not None:
            recent_buys = self.history_df[
                (self.history_df["action_type"].isin(["Buy", "Add"])) &
                (self.history_df["period"] == "Q4 2024")
            ].groupby("ticker")["manager_id"].nunique()
            
            multi_manager = multi_manager.merge(
                recent_buys.rename("recent_buyers"),
                left_index=True, right_index=True, how="left"
            )
            multi_manager["recent_buyers"] = multi_manager["recent_buyers"].fillna(0)
        
        # Add manager names
        multi_manager["manager_names"] = multi_manager["managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")[:10]])
        )
        
        # Sort by consensus score
        multi_manager = multi_manager.sort_values("consensus_score", ascending=False)
        
        return self.apply_precision_formatting(multi_manager.reset_index()).head(50)
    
    def analyze_stocks_by_price(self, max_price: float, label: str) -> pd.DataFrame:
        """Analyze stocks under a specific price threshold."""
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # We need current price - check if it's in holdings or stocks data
        analysis_df = self.holdings_df.copy()
        
        # Try to get current price
        if "current_price" not in analysis_df.columns:
            if self.stocks_df is not None and "current_price" in self.stocks_df.columns:
                analysis_df = analysis_df.merge(
                    self.stocks_df[["ticker", "current_price", "market_cap", "sector"]],
                    on="ticker", how="left"
                )
            else:
                # Estimate from value/shares if no price data
                analysis_df["current_price"] = analysis_df["value"] / analysis_df["shares"]
        
        # Filter by price and exclude positions with 0.0 value (historical positions)
        price_filtered = analysis_df[
            (analysis_df["current_price"] > 0) & 
            (analysis_df["current_price"] <= max_price) &
            (analysis_df["value"] > 0.0)  # Filter out historical positions
        ].copy()
        
        if price_filtered.empty:
            return pd.DataFrame()
        
        # Group by ticker
        grouped = price_filtered.groupby("ticker").agg({
            "manager_id": ["count", self.get_manager_list],
            "current_price": "first",
            "shares": "sum",
            "value": "sum",
            "portfolio_percent": ["mean", "max"],
        })
        
        grouped.columns = [
            "manager_count", "managers", "current_price", "total_shares",
            "total_value", "avg_portfolio_pct", "max_portfolio_pct"
        ]
        
        # Add stock info if not already there
        if "market_cap" not in grouped.columns and self.stocks_df is not None:
            stock_info = self.stocks_df[["ticker", "market_cap", "sector"]].copy()
            grouped = grouped.merge(stock_info, left_index=True, right_on="ticker", how="left")
            grouped = grouped.set_index("ticker")
        
        # Add recent activity
        if self.history_df is not None:
            recent_activity = self.history_df[
                self.history_df["action_type"].isin(["Buy", "Add"])
            ].groupby("ticker").agg({
                "period": ["count", "max"]
            })
            recent_activity.columns = ["buy_count", "last_buy_period"]
            grouped = grouped.merge(recent_activity, left_index=True, right_index=True, how="left")
        
        # Add manager names
        grouped["manager_names"] = grouped["managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
        )
        
        # Filter out any remaining 0.0 total_value entries
        grouped = grouped[grouped["total_value"] > 0.0].copy()
        
        # Sort by manager count and portfolio percentage
        grouped = grouped.sort_values(
            ["manager_count", "avg_portfolio_pct"], ascending=[False, False]
        )
        
        return self.apply_precision_formatting(grouped.reset_index()).head(50)
    
    def analyze_stocks_by_market_cap(self, cap_category: str) -> pd.DataFrame:
        """Analyze stocks by market cap category."""
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # Get market cap data
        analysis_df = self.holdings_df.copy()
        
        if "market_cap" not in analysis_df.columns and self.stocks_df is not None:
            analysis_df = analysis_df.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                on="ticker", how="left"
            )
        
        # Categorize market caps
        analysis_df["cap_category"] = analysis_df["market_cap"].apply(self.categorize_market_cap)
        
        # Filter by category
        cap_filtered = analysis_df[analysis_df["cap_category"] == cap_category].copy()
        
        if cap_filtered.empty:
            return pd.DataFrame()
        
        # Group by ticker
        grouped = cap_filtered.groupby("ticker").agg({
            "manager_id": ["count", self.get_manager_list],
            "market_cap": "first",
            "shares": "sum",
            "value": "sum",
            "portfolio_percent": ["mean", "max"],
        })
        
        grouped.columns = [
            "manager_count", "managers", "market_cap", "total_shares",
            "total_value", "avg_portfolio_pct", "max_portfolio_pct"
        ]
        
        # Format market cap
        grouped["market_cap_display"] = grouped["market_cap"].apply(self.format_market_cap)
        
        # Add stock info
        if "sector" not in grouped.columns and self.stocks_df is not None:
            stock_info = self.stocks_df[["ticker", "sector"]].copy()
            grouped = grouped.merge(stock_info, left_index=True, right_on="ticker", how="left")
            grouped = grouped.set_index("ticker")
        
        # Add recent activity
        if self.history_df is not None:
            recent_activity = self.history_df[
                self.history_df["action_type"].isin(["Buy", "Add"])
            ].groupby("ticker").agg({
                "period": ["count", "max"]
            })
            recent_activity.columns = ["buy_count", "last_buy_period"]
            grouped = grouped.merge(recent_activity, left_index=True, right_index=True, how="left")
        
        # Add manager names
        grouped["manager_names"] = grouped["managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
        )
        
        # Sort by total value
        grouped = grouped.sort_values("total_value", ascending=False)
        
        return self.apply_precision_formatting(grouped.reset_index()).head(50)
    
    def analyze_52_week_low_buys(self) -> pd.DataFrame:
        """Find stocks being bought near 52-week lows."""
        if self.history_df is None:
            return pd.DataFrame()
        
        # Get recent buy activities
        recent_buys = self.history_df[
            (self.history_df["action_type"].isin(["Buy", "Add"])) &
            (self.history_df["period"].isin(["Q4 2024", "Q3 2024"]))
        ].copy()
        
        if recent_buys.empty:
            return pd.DataFrame()
        
        # Group by ticker
        buy_summary = recent_buys.groupby("ticker").agg({
            "manager_id": ["count", lambda x: ", ".join(x.unique())],
            "shares": "sum",
            "period": lambda x: ", ".join(x.unique()),
        })
        
        buy_summary.columns = ["buy_count", "buying_managers", "total_shares_bought", "periods"]
        
        # Get current holdings for these stocks
        if self.holdings_df is not None:
            current_data = self.holdings_df.groupby("ticker").agg({
                "value": "sum",
                "portfolio_percent": "mean",
            })
            buy_summary = buy_summary.merge(current_data, left_index=True, right_index=True, how="left")
        
        # Add stock info
        if self.stocks_df is not None:
            buy_summary = buy_summary.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Add manager names
        buy_summary["buying_manager_names"] = buy_summary["buying_managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
        )
        
        # Note: Without actual 52-week low data, we're showing all recent buys
        buy_summary["buy_signal"] = "Recent accumulation"
        
        # Sort by buy count
        buy_summary = buy_summary.sort_values("buy_count", ascending=False)
        
        return self.apply_precision_formatting(buy_summary.reset_index()).head(50)
    
    def analyze_52_week_high_sells(self) -> pd.DataFrame:
        """Find stocks being sold near 52-week highs."""
        if self.history_df is None:
            return pd.DataFrame()
        
        # Get recent sell activities
        recent_sells = self.history_df[
            (self.history_df["action_type"].isin(["Sell", "Reduce"])) &
            (self.history_df["period"].isin(["Q4 2024", "Q3 2024"]))
        ].copy()
        
        if recent_sells.empty:
            return pd.DataFrame()
        
        # Group by ticker
        sell_summary = recent_sells.groupby("ticker").agg({
            "manager_id": ["count", lambda x: ", ".join(x.unique())],
            "shares": "sum",
            "period": lambda x: ", ".join(x.unique()),
        })
        
        sell_summary.columns = ["sell_count", "selling_managers", "total_shares_sold", "periods"]
        
        # Get current holdings for these stocks
        if self.holdings_df is not None:
            current_data = self.holdings_df.groupby("ticker").agg({
                "value": "sum",
                "portfolio_percent": "mean",
            })
            sell_summary = sell_summary.merge(current_data, left_index=True, right_index=True, how="left")
        
        # Add stock info
        if self.stocks_df is not None:
            sell_summary = sell_summary.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Add manager names
        sell_summary["selling_manager_names"] = sell_summary["selling_managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
        )
        
        # Note: Without actual 52-week high data, we're showing all recent sells
        sell_summary["sell_signal"] = "Profit taking"
        
        # Sort by sell count
        sell_summary = sell_summary.sort_values("sell_count", ascending=False)
        
        return self.apply_precision_formatting(sell_summary.reset_index()).head(50)
    
    def analyze_most_sold_stocks(self) -> pd.DataFrame:
        """Find stocks with the most selling activity."""
        if self.history_df is None:
            return pd.DataFrame()
        
        # Get all sell activities
        all_sells = self.history_df[
            self.history_df["action_type"].isin(["Sell", "Reduce", "Sold Out"])
        ].copy()
        
        if all_sells.empty:
            return pd.DataFrame()
        
        # Group by ticker
        sell_analysis = all_sells.groupby("ticker").agg({
            "manager_id": ["count", lambda x: ", ".join(x.unique())],
            "action_type": lambda x: dict(x.value_counts()),
            "period": lambda x: ", ".join(x.unique()),
        })
        
        sell_analysis.columns = ["total_sells", "selling_managers", "action_breakdown", "periods"]
        
        # Get current holdings if any remain
        if self.holdings_df is not None:
            current_holders = self.holdings_df.groupby("ticker")["manager_id"].nunique()
            sell_analysis = sell_analysis.merge(
                current_holders.rename("remaining_holders"),
                left_index=True, right_index=True, how="left"
            )
            sell_analysis["remaining_holders"] = sell_analysis["remaining_holders"].fillna(0)
        
        # Add stock info
        if self.stocks_df is not None:
            sell_analysis = sell_analysis.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Calculate exit percentage
        sell_analysis["exit_rate"] = (
            sell_analysis["total_sells"] / 
            (sell_analysis["total_sells"] + sell_analysis.get("remaining_holders", 0))
        ).fillna(1.0) * 100
        
        # Format action breakdown
        sell_analysis["sell_pattern"] = sell_analysis["action_breakdown"].apply(
            lambda x: ", ".join([f"{k}: {v}" for k, v in x.items()])
        )
        
        # Add manager names
        sell_analysis["selling_manager_names"] = sell_analysis["selling_managers"].apply(
            lambda x: ", ".join([self.manager_name_map.get(m.strip(), m.strip()) for m in x.split(",")])
        )
        
        # Sort by total sells
        sell_analysis = sell_analysis.sort_values("total_sells", ascending=False)
        
        return self.apply_precision_formatting(sell_analysis.reset_index()).head(50)
    
    def analyze_contrarian_plays(self) -> pd.DataFrame:
        """Find stocks with mixed buy/sell signals."""
        if self.history_df is None:
            return pd.DataFrame()
        
        # Get recent activities
        recent_activities = self.history_df[
            self.history_df["period"].isin(["Q4 2024", "Q3 2024"])
        ].copy()
        
        if recent_activities.empty:
            return pd.DataFrame()
        
        # Count buys and sells by ticker
        activity_summary = recent_activities.groupby("ticker").agg({
            "action_type": lambda x: dict(x.value_counts()),
            "manager_id": lambda x: len(x.unique()),
            "period": lambda x: ", ".join(x.unique()),
        })
        
        activity_summary.columns = ["action_breakdown", "active_managers", "periods"]
        
        # Calculate buy and sell counts
        activity_summary["buy_actions"] = activity_summary["action_breakdown"].apply(
            lambda x: x.get("Buy", 0) + x.get("Add", 0)
        )
        activity_summary["sell_actions"] = activity_summary["action_breakdown"].apply(
            lambda x: x.get("Sell", 0) + x.get("Reduce", 0) + x.get("Sold Out", 0)
        )
        
        # Filter for mixed signals (both buys and sells)
        contrarian = activity_summary[
            (activity_summary["buy_actions"] > 0) & 
            (activity_summary["sell_actions"] > 0)
        ].copy()
        
        if contrarian.empty:
            return pd.DataFrame()
        
        # Calculate contrarian score
        contrarian["contrarian_score"] = (
            contrarian["buy_actions"] - contrarian["sell_actions"]
        ) / contrarian["active_managers"]
        
        # Get current data
        if self.holdings_df is not None:
            current_data = self.holdings_df.groupby("ticker").agg({
                "manager_id": "nunique",
                "value": "sum",
                "portfolio_percent": "mean",
            })
            current_data.columns = ["current_holders", "total_value", "avg_portfolio_pct"]
            contrarian = contrarian.merge(current_data, left_index=True, right_index=True, how="left")
        
        # Add stock info
        if self.stocks_df is not None:
            contrarian = contrarian.merge(
                self.stocks_df[["ticker", "market_cap", "pe_ratio", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Categorize contrarian type
        contrarian["signal"] = contrarian["contrarian_score"].apply(
            lambda x: "Net Buying" if x > 0 else "Net Selling" if x < 0 else "Balanced"
        )
        
        # Sort by absolute contrarian score
        contrarian = contrarian.sort_values("contrarian_score", key=abs, ascending=False)
        
        return self.apply_precision_formatting(contrarian.reset_index()).head(50)
    
    def analyze_concentration_changes(self) -> pd.DataFrame:
        """Find stocks with significant portfolio concentration changes."""
        if self.history_df is None or self.holdings_df is None:
            return pd.DataFrame()
        
        # Get recent concentration changes from activity
        recent_changes = self.history_df[
            (self.history_df["period"].isin(["Q4 2024", "Q3 2024"])) &
            (self.history_df["percentage_change"].notna())
        ].copy()
        
        if recent_changes.empty:
            return pd.DataFrame()
        
        # Get significant changes (>10% position change)
        significant_changes = recent_changes[
            abs(recent_changes["percentage_change"]) > 10
        ].copy()
        
        if significant_changes.empty:
            return pd.DataFrame()
        
        # Group by ticker and manager
        change_summary = significant_changes.groupby(["ticker", "manager_id"]).agg({
            "percentage_change": "sum",
            "action_type": "first",
            "period": "first",
        }).reset_index()
        
        # Get current positions
        current_positions = self.holdings_df[["ticker", "manager_id", "portfolio_percent", "value"]]
        
        # Merge with current data
        change_summary = change_summary.merge(
            current_positions, on=["ticker", "manager_id"], how="left"
        )
        
        # Add manager names
        change_summary["manager_name"] = change_summary["manager_id"].map(
            lambda x: self.manager_name_map.get(x, x)
        )
        
        # Group by ticker for overview
        ticker_summary = change_summary.groupby("ticker").agg({
            "manager_id": "count",
            "percentage_change": ["mean", "max", "min"],
            "value": "sum",
            "portfolio_percent": "mean",
        })
        
        ticker_summary.columns = [
            "managers_changing", "avg_change_pct", "max_increase_pct", 
            "max_decrease_pct", "total_value", "avg_portfolio_pct"
        ]
        
        # Add stock info
        if self.stocks_df is not None:
            ticker_summary = ticker_summary.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Add details of who's changing positions
        change_details = change_summary.groupby("ticker").apply(
            lambda x: ", ".join([
                f"{row['manager_name']}: {row['percentage_change']:+.1f}%"
                for _, row in x.nlargest(3, "percentage_change", keep="all").iterrows()
            ])
        )
        ticker_summary["top_changes"] = change_details
        
        # Calculate momentum
        ticker_summary["concentration_momentum"] = (
            ticker_summary["max_increase_pct"] + ticker_summary["max_decrease_pct"]
        )
        
        # Filter out positions with 0.0 total_value (no longer held)
        current_holdings_only = ticker_summary[ticker_summary["total_value"] > 0.0].copy()
        
        # Sort by absolute average change
        current_holdings_only = current_holdings_only.sort_values(
            "avg_change_pct", key=abs, ascending=False
        )
        
        return self.apply_precision_formatting(current_holdings_only.reset_index()).head(50)
    
    def analyze_historical_concentration_changes(self) -> pd.DataFrame:
        """Find historical concentration changes for positions no longer held."""
        if self.history_df is None or self.holdings_df is None:
            return pd.DataFrame()
        
        # Get recent concentration changes from activity
        recent_changes = self.history_df[
            (self.history_df["period"].isin(["Q4 2024", "Q3 2024"])) &
            (self.history_df["percentage_change"].notna())
        ].copy()
        
        if recent_changes.empty:
            return pd.DataFrame()
        
        # Get significant changes (>10% position change)
        significant_changes = recent_changes[
            abs(recent_changes["percentage_change"]) > 10
        ].copy()
        
        if significant_changes.empty:
            return pd.DataFrame()
        
        # Group by ticker and manager
        change_summary = significant_changes.groupby(["ticker", "manager_id"]).agg({
            "percentage_change": "sum",
            "action_type": "first",
            "period": "first",
        }).reset_index()
        
        # Get current positions
        current_positions = self.holdings_df[["ticker", "manager_id", "portfolio_percent", "value"]]
        
        # Merge with current data
        change_summary = change_summary.merge(
            current_positions, on=["ticker", "manager_id"], how="left"
        )
        
        # Add manager names
        change_summary["manager_name"] = change_summary["manager_id"].map(
            lambda x: self.manager_name_map.get(x, x)
        )
        
        # Group by ticker for overview
        ticker_summary = change_summary.groupby("ticker").agg({
            "manager_id": "count",
            "percentage_change": ["mean", "max", "min"],
            "value": "sum",
            "portfolio_percent": "mean",
        })
        
        ticker_summary.columns = [
            "managers_changing", "avg_change_pct", "max_increase_pct", 
            "max_decrease_pct", "total_value", "avg_portfolio_pct"
        ]
        
        # Add stock info
        if self.stocks_df is not None:
            ticker_summary = ticker_summary.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                left_index=True, right_on="ticker", how="left"
            ).set_index("ticker")
        
        # Add details of who's changing positions
        change_details = change_summary.groupby("ticker").apply(
            lambda x: ", ".join([
                f"{row['manager_name']}: {row['percentage_change']:+.1f}%"
                for _, row in x.nlargest(3, "percentage_change", keep="all").iterrows()
            ])
        )
        ticker_summary["top_changes"] = change_details
        
        # Calculate momentum
        ticker_summary["concentration_momentum"] = (
            ticker_summary["max_increase_pct"] + ticker_summary["max_decrease_pct"]
        )
        
        # Filter for historical positions only (0.0 total_value - no longer held)
        historical_only = ticker_summary[ticker_summary["total_value"] == 0.0].copy()
        
        # Sort by absolute average change
        historical_only = historical_only.sort_values(
            "avg_change_pct", key=abs, ascending=False
        )
        
        return self.apply_precision_formatting(historical_only.reset_index()).head(50)

    def analyze_sector_rotation(self) -> pd.DataFrame:
        """Analyze sector rotation patterns."""
        if self.holdings_df is None:
            return pd.DataFrame()
        
        # Get sector data
        if "sector" in self.holdings_df.columns:
            sector_df = self.holdings_df.copy()
        elif self.stocks_df is not None and "sector" in self.stocks_df.columns:
            sector_df = self.holdings_df.merge(
                self.stocks_df[["ticker", "sector"]], on="ticker", how="left"
            )
        else:
            return pd.DataFrame()
        
        # Filter out unknown sectors
        sector_df = sector_df[sector_df["sector"].notna()]
        
        if sector_df.empty:
            return pd.DataFrame()
        
        # Current sector allocation
        current_allocation = sector_df.groupby("sector").agg({
            "value": "sum",
            "manager_id": "nunique",
            "ticker": "nunique",
        })
        current_allocation.columns = ["total_value", "manager_count", "stock_count"]
        
        # Add historical activity if available
        if self.history_df is not None:
            # Merge sector info with history
            history_with_sector = self.history_df.merge(
                self.stocks_df[["ticker", "sector"]], on="ticker", how="left"
            )
            
            # Recent sector activity
            recent_activity = history_with_sector[
                history_with_sector["period"].isin(["Q4 2024", "Q3 2024"])
            ].groupby("sector").agg({
                "action_type": lambda x: dict(x.value_counts()),
                "manager_id": "nunique",
                "period": lambda x: ", ".join(x.unique()),
            })
            recent_activity.columns = ["activity_breakdown", "active_managers", "periods"]
            
            # Calculate net activity
            recent_activity["buy_actions"] = recent_activity["activity_breakdown"].apply(
                lambda x: x.get("Buy", 0) + x.get("Add", 0)
            )
            recent_activity["sell_actions"] = recent_activity["activity_breakdown"].apply(
                lambda x: x.get("Sell", 0) + x.get("Reduce", 0) + x.get("Sold Out", 0)
            )
            recent_activity["net_activity"] = (
                recent_activity["buy_actions"] - recent_activity["sell_actions"]
            )
            
            # Merge with current allocation
            sector_rotation = current_allocation.merge(
                recent_activity[["net_activity", "buy_actions", "sell_actions", "active_managers", "periods"]],
                left_index=True, right_index=True, how="left"
            )
        else:
            sector_rotation = current_allocation
        
        # Calculate sector weight
        total_value = sector_rotation["total_value"].sum()
        sector_rotation["sector_weight_pct"] = (
            sector_rotation["total_value"] / total_value * 100
        )
        
        # Add rotation signal
        if "net_activity" in sector_rotation.columns:
            sector_rotation["rotation_signal"] = sector_rotation["net_activity"].apply(
                lambda x: "Inflow" if x > 5 else "Outflow" if x < -5 else "Stable"
            )
            
            # Calculate rotation score
            sector_rotation["rotation_score"] = (
                sector_rotation["net_activity"] / sector_rotation["manager_count"]
            )
        
        # Sort by sector weight
        sector_rotation = sector_rotation.sort_values("total_value", ascending=False)
        
        return self.apply_precision_formatting(sector_rotation).head(20)
    
    def analyze_highest_portfolio_concentration(self) -> pd.DataFrame:
        """Find stocks with highest portfolio concentration."""
        if self.holdings_df is None:
            raise ValueError("Holdings data not loaded")
        
        # Get top positions by portfolio percentage
        top_positions = self.holdings_df.nlargest(100, "portfolio_percent").copy()
        
        # Add manager names
        top_positions["manager_name"] = top_positions["manager_id"].map(
            lambda x: self.manager_name_map.get(x, x)
        )
        
        # Add stock info if available
        if self.stocks_df is not None:
            top_positions = top_positions.merge(
                self.stocks_df[["ticker", "market_cap", "sector"]],
                on="ticker", how="left"
            )
        
        # Add holding period if available
        if self.history_df is not None:
            first_buys = self.history_df[
                self.history_df["action_type"] == "Buy"
            ].groupby(["ticker", "manager_id"])["period"].first().reset_index()
            first_buys.columns = ["ticker", "manager_id", "first_buy_period"]
            
            top_positions = top_positions.merge(
                first_buys, on=["ticker", "manager_id"], how="left"
            )
        
        # Calculate concentration risk
        top_positions["concentration_risk"] = top_positions["portfolio_percent"].apply(
            lambda x: "Very High" if x > 20 else "High" if x > 10 else "Moderate" if x > 5 else "Low"
        )
        
        # Select and order columns
        columns = [
            "ticker", "manager_name", "portfolio_percent", "value", "shares",
            "concentration_risk"
        ]
        
        # Skip name column as it doesn't exist in stocks data
        if "first_buy_period" in top_positions.columns:
            columns.append("first_buy_period")
        if "sector" in top_positions.columns:
            columns.append("sector")
        
        result = top_positions[columns].copy()
        
        # Sort by portfolio percentage
        result = result.sort_values("portfolio_percent", ascending=False)
        
        return self.apply_precision_formatting(result).head(50)

    def create_sector_distribution_chart(self) -> None:
        """Create and save sector distribution visualization."""
        sector_data = self.analyze_sector_distribution()

        if len(sector_data) > 0:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

            # Pie chart for value distribution
            top_sectors = sector_data.head(10)
            ax1.pie(
                top_sectors["total_value"],
                labels=top_sectors["sector"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax1.set_title("Portfolio Value by Sector (Top 10)")

            # Bar chart for stock count
            ax2.bar(
                range(len(top_sectors)),
                top_sectors["unique_stocks"],
                color="skyblue",
            )
            ax2.set_xticks(range(len(top_sectors)))
            ax2.set_xticklabels(top_sectors["sector"], rotation=45, ha="right")
            ax2.set_ylabel("Number of Unique Stocks")
            ax2.set_title("Stock Count by Sector")

            plt.tight_layout()
            plt.savefig(
                self.visual_dir / "sector_distribution.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            logging.info("Created sector distribution chart")

    def create_manager_performance_chart(self) -> None:
        """Create and save manager performance visualization."""
        manager_data = self.analyze_manager_performance()

        if len(manager_data) > 0:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Scatter plot
            top_managers = manager_data.head(20)
            ax.scatter(
                top_managers["position_count"],
                top_managers["total_value"] / 1e9,  # Convert to billions
                s=top_managers["concentration_score"] * 100,
                alpha=0.6,
                c=range(len(top_managers)),
                cmap="viridis",
            )

            # Add labels
            for _, row in top_managers.iterrows():
                ax.annotate(
                    row["manager_name"][:15],  # Truncate long names
                    (row["position_count"], row["total_value"] / 1e9),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            ax.set_xlabel("Number of Positions")
            ax.set_ylabel("Total Portfolio Value ($B)")
            ax.set_title(
                "Manager Portfolio Analysis\n(Bubble size = Concentration Score)"
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.visual_dir / "manager_performance.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            logging.info("Created manager performance chart")

    def create_top_holdings_chart(self) -> None:
        """Create and save top holdings visualization."""
        top_holdings = self.analyze_top_holdings()

        if len(top_holdings) > 0:
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))

            # Horizontal bar chart
            top_20 = top_holdings.head(20)
            y_pos = np.arange(len(top_20))

            ax.barh(
                y_pos,
                top_20["manager_count"],
                color=plt.cm.Blues(
                    top_20["manager_count"] / top_20["manager_count"].max()
                ),
            )

            # Add value labels
            for i, (count, value) in enumerate(
                zip(top_20["manager_count"], top_20["total_value"])
            ):
                ax.text(
                    count + 0.1,
                    i,
                    f"${value / 1e9:.1f}B",
                    va="center",
                    fontsize=8,
                )

            # Customize chart
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [
                    f"{row['ticker']} - "
                    f"{row.get('name', row.get('stock', row['ticker']))[:30]}"
                    for _, row in top_20.iterrows()
                ]
            )
            ax.invert_yaxis()
            ax.set_xlabel("Number of Managers Holding")
            ax.set_title("Top 20 Holdings by Manager Count")
            ax.grid(True, axis="x", alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.visual_dir / "top_holdings_by_managers.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            logging.info("Created top holdings chart")

    def analyze_manager_performance_3year(self) -> pd.DataFrame:
        """Analyze 3-year manager performance with portfolio evolution and returns."""
        if self.history_df is None:
            logging.warning("No history data available for 3-year analysis")
            return pd.DataFrame()
        
        # Calculate 3-year lookback from latest date
        latest_date = pd.to_datetime(self.history_df["date"]).max()
        three_years_ago = latest_date - pd.DateOffset(years=3)
        
        # Filter to 3-year period
        three_year_data = self.history_df[
            pd.to_datetime(self.history_df["date"]) >= three_years_ago
        ].copy()
        
        if three_year_data.empty:
            logging.warning("No data available for 3-year analysis period")
            return pd.DataFrame()
        
        # Group by manager and calculate metrics
        manager_performance = []
        
        for manager_id in three_year_data["manager_id"].unique():
            manager_data = three_year_data[three_year_data["manager_id"] == manager_id]
            
            # Get manager name
            manager_name = self.manager_name_map.get(manager_id, manager_id)
            
            # Calculate portfolio value evolution with proper date handling
            manager_data_with_dates = manager_data.copy()
            manager_data_with_dates["date"] = pd.to_datetime(manager_data_with_dates["date"])
            manager_periods = manager_data_with_dates.groupby("date")["value"].sum().sort_index()
            
            if len(manager_periods) < 2:
                continue
                
            start_value = manager_periods.iloc[0]
            end_value = manager_periods.iloc[-1]
            
            # Calculate total return
            total_return_pct = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0
            
            # Count major activities
            buys = len(manager_data[manager_data["action_class"] == "buy"])
            sells = len(manager_data[manager_data["action_class"] == "sell"])
            
            # Get top 5 biggest plays by portfolio impact
            top_plays = manager_data.nlargest(5, "portfolio_impact")[
                ["ticker", "action", "portfolio_impact", "date"]
            ]
            top_plays_str = "; ".join([
                f"{row['ticker']} ({row['action']}, {row['portfolio_impact']:.1f}%, {row['date']})"
                for _, row in top_plays.iterrows()
            ])
            
            # Calculate volatility (std of quarterly values)
            quarterly_count = 0
            try:
                quarterly_values = manager_periods.resample("QE").last().dropna()
                volatility = quarterly_values.std() if len(quarterly_values) > 1 else 0
                quarterly_count = len(quarterly_values)
            except (TypeError, ValueError):
                # Fallback if resample fails
                volatility = manager_periods.std() if len(manager_periods) > 1 else 0
                quarterly_count = len(manager_periods)
            
            # Average position size
            avg_position_impact = manager_data["portfolio_impact"].mean()
            
            manager_performance.append({
                "manager_id": manager_id,
                "manager_name": manager_name,
                "start_portfolio_value": start_value,
                "end_portfolio_value": end_value,
                "total_return_pct": total_return_pct,
                "volatility": volatility,
                "total_transactions": len(manager_data),
                "buy_transactions": buys,
                "sell_transactions": sells,
                "avg_position_impact": avg_position_impact,
                "top_5_plays": top_plays_str,
                "period_start": manager_periods.index[0],
                "period_end": manager_periods.index[-1],
                "quarters_active": quarterly_count
            })
        
        performance_df = pd.DataFrame(manager_performance)
        
        if not performance_df.empty:
            # Sort by total return
            performance_df = performance_df.sort_values("total_return_pct", ascending=False)
            
            # Save to historical folder
            historical_dir = Path(self.output_dir) / "historical"
            historical_dir.mkdir(exist_ok=True)
            
            performance_df.round(2).to_csv(
                historical_dir / "manager_performance_3year_summary.csv", index=False
            )
            
            logging.info(f"Created 3-year manager performance analysis for {len(performance_df)} managers")
        
        return performance_df

    def analyze_quarterly_activity_timeline(self) -> pd.DataFrame:
        """Create detailed quarterly activity timeline with manager names and major changes."""
        if self.history_df is None:
            logging.warning("No history data available for quarterly timeline")
            return pd.DataFrame()
        
        # Convert dates and filter to significant activities (>1% portfolio impact)
        timeline_data = self.history_df[self.history_df["portfolio_impact"].abs() >= 1.0].copy()
        timeline_data["date"] = pd.to_datetime(timeline_data["date"])
        timeline_data["quarter"] = timeline_data["date"].dt.to_period("Q")
        
        # Group by quarter and summarize major activities
        quarterly_summary = []
        
        for quarter in sorted(timeline_data["quarter"].unique()):
            quarter_data = timeline_data[timeline_data["quarter"] == quarter]
            
            # Major buys (top 5 by portfolio impact)
            major_buys = quarter_data[quarter_data["action_class"] == "buy"].nlargest(5, "portfolio_impact")
            major_sells = quarter_data[quarter_data["action_class"] == "sell"].nlargest(5, "portfolio_impact")
            
            # Active managers this quarter
            active_managers = quarter_data["manager_id"].nunique()
            manager_names = ", ".join([
                self.manager_name_map.get(mid, mid) 
                for mid in quarter_data["manager_id"].unique()[:10]  # Top 10 most active
            ])
            
            # Summary stats
            total_transactions = len(quarter_data)
            avg_impact = quarter_data["portfolio_impact"].mean()
            
            # Format major activities
            buy_summary = "; ".join([
                f"{self.manager_name_map.get(row['manager_id'], row['manager_id'])}: {row['ticker']} +{row['portfolio_impact']:.1f}%"
                for _, row in major_buys.iterrows()
            ])
            
            sell_summary = "; ".join([
                f"{self.manager_name_map.get(row['manager_id'], row['manager_id'])}: {row['ticker']} {row['portfolio_impact']:.1f}%"
                for _, row in major_sells.iterrows()
            ])
            
            quarterly_summary.append({
                "quarter": str(quarter),
                "year": quarter.year,
                "quarter_num": quarter.quarter,
                "active_managers": active_managers,
                "total_transactions": total_transactions,
                "avg_portfolio_impact": avg_impact,
                "major_buys": buy_summary,
                "major_sells": sell_summary,
                "top_active_managers": manager_names
            })
        
        timeline_df = pd.DataFrame(quarterly_summary)
        
        if not timeline_df.empty:
            # Sort by quarter
            timeline_df = timeline_df.sort_values(["year", "quarter_num"], ascending=False)
            
            # Save to historical folder
            historical_dir = Path(self.output_dir) / "historical"
            historical_dir.mkdir(exist_ok=True)
            
            timeline_df.round(2).to_csv(
                historical_dir / "quarterly_activity_timeline.csv", index=False
            )
            
            logging.info(f"Created quarterly activity timeline for {len(timeline_df)} quarters")
        
        return timeline_df

    def analyze_top_performing_plays(self) -> pd.DataFrame:
        """Analyze the highest performing individual plays across all managers and time periods."""
        if self.history_df is None:
            logging.warning("No history data available for top plays analysis")
            return pd.DataFrame()
        
        # Filter to buy actions only (we want to see the best stock picks)
        buy_actions = self.history_df[self.history_df["action_class"] == "buy"].copy()
        
        if buy_actions.empty:
            return pd.DataFrame()
        
        # For each buy, try to find subsequent performance
        # We'll use portfolio_impact as a proxy for conviction/size
        top_plays = []
        
        # Get the biggest buys (top 100 by portfolio impact)
        major_buys = buy_actions.nlargest(100, "portfolio_impact")
        
        for _, buy_row in major_buys.iterrows():
            manager_id = buy_row["manager_id"]
            ticker = buy_row["ticker"]
            buy_date = buy_row["date"]
            
            # Find if this position was later sold
            manager_history = self.history_df[
                (self.history_df["manager_id"] == manager_id) &
                (self.history_df["ticker"] == ticker) &
                (pd.to_datetime(self.history_df["date"]) > pd.to_datetime(buy_date))
            ]
            
            # Check if there was a sell action
            sell_actions = manager_history[manager_history["action_class"] == "sell"]
            
            if not sell_actions.empty:
                # Find the largest sell action (position exit)
                major_sell = sell_actions.loc[sell_actions["portfolio_impact"].abs().idxmax()]
                
                # Calculate holding period
                hold_days = (pd.to_datetime(major_sell["date"]) - pd.to_datetime(buy_date)).days
                
                # Estimate return based on portfolio impact changes
                # This is approximate since we don't have exact prices
                buy_impact = buy_row["portfolio_impact"]
                sell_impact = abs(major_sell["portfolio_impact"])
                estimated_return = ((sell_impact - buy_impact) / buy_impact * 100) if buy_impact > 0 else 0
            else:
                # Position still held or no clear exit
                hold_days = (pd.to_datetime("today") - pd.to_datetime(buy_date)).days
                estimated_return = None
            
            top_plays.append({
                "manager_id": manager_id,
                "manager_name": self.manager_name_map.get(manager_id, manager_id),
                "ticker": ticker,
                "company": buy_row.get("company", ""),
                "buy_date": buy_date,
                "sell_date": major_sell["date"] if not sell_actions.empty else None,
                "holding_days": hold_days,
                "portfolio_impact_buy": buy_row["portfolio_impact"],
                "portfolio_impact_sell": abs(major_sell["portfolio_impact"]) if not sell_actions.empty else None,
                "estimated_return_pct": estimated_return,
                "buy_action": buy_row["action"],
                "sell_action": major_sell["action"] if not sell_actions.empty else "Still Held"
            })
        
        plays_df = pd.DataFrame(top_plays)
        
        if not plays_df.empty:
            # Sort by estimated return (nulls last)
            plays_df = plays_df.sort_values("estimated_return_pct", ascending=False, na_position="last")
            
            # Save to historical folder
            historical_dir = Path(self.output_dir) / "historical"
            historical_dir.mkdir(exist_ok=True)
            
            plays_df.round(2).to_csv(
                historical_dir / "top_performing_plays.csv", index=False
            )
            
            logging.info(f"Created top performing plays analysis for {len(plays_df)} major positions")
        
        return plays_df

    def create_historical_visualizations(self) -> None:
        """Create comprehensive visualizations for historical analysis."""
        historical_dir = Path(self.output_dir) / "historical"
        visuals_dir = historical_dir / "visuals"
        visuals_dir.mkdir(exist_ok=True)
        
        # Set up the visual style
        plt.style.use('default')
        sns.set_palette("husl")
        
        try:
            # 1. Manager Performance 3-Year Returns Chart
            self._create_manager_performance_3year_chart(visuals_dir)
            
            # 2. Quarterly Activity Timeline Chart
            self._create_quarterly_activity_chart(visuals_dir)
            
            # 3. Top Performing Plays Chart
            self._create_top_plays_chart(visuals_dir)
            
            # 4. Manager Performance Distribution Chart
            self._create_performance_distribution_chart(visuals_dir)
            
            logging.info("Created comprehensive historical visualizations")
            
        except Exception as e:
            logging.error(f"Error creating historical visualizations: {e}")

    def _create_manager_performance_3year_chart(self, visuals_dir: Path) -> None:
        """Create 3-year manager performance visualization."""
        # Load the 3-year performance data
        performance_file = Path(self.output_dir) / "historical" / "manager_performance_3year_summary.csv"
        if not performance_file.exists():
            return
        
        df = pd.read_csv(performance_file)
        if df.empty:
            return
        
        # Create a comprehensive performance chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('3-Year Manager Performance Analysis', fontsize=20, fontweight='bold')
        
        # Top 15 performers by return
        top_performers = df.head(15).copy()
        
        # 1. Total Returns Bar Chart
        bars = ax1.barh(top_performers['manager_name'], top_performers['total_return_pct'], 
                       color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_performers))))
        ax1.set_xlabel('Total Return (%)', fontsize=12)
        ax1.set_title('Top 15 Manager Returns (3-Year)', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, top_performers['total_return_pct']):
            ax1.text(bar.get_width() + max(top_performers['total_return_pct']) * 0.01, 
                    bar.get_y() + bar.get_height()/2, f'{value:.1f}%', 
                    ha='left', va='center', fontweight='bold')
        
        # 2. Portfolio Value Evolution (Start vs End)
        x_pos = np.arange(len(top_performers))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, top_performers['start_portfolio_value'], width, 
                       label='Start Value', alpha=0.8, color='lightcoral')
        bars2 = ax2.bar(x_pos + width/2, top_performers['end_portfolio_value'], width,
                       label='End Value', alpha=0.8, color='lightgreen')
        
        ax2.set_xlabel('Managers', fontsize=12)
        ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax2.set_title('Portfolio Value Evolution', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(top_performers['manager_name'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Transaction Activity vs Returns Scatter
        ax3.scatter(df['total_transactions'], df['total_return_pct'], 
                   alpha=0.6, s=60, c=df['total_return_pct'], cmap='RdYlGn')
        ax3.set_xlabel('Total Transactions', fontsize=12)
        ax3.set_ylabel('Total Return (%)', fontsize=12)
        ax3.set_title('Activity vs Performance', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Add labels for top performers
        for _, row in top_performers.head(5).iterrows():
            ax3.annotate(row['manager_name'], 
                        (row['total_transactions'], row['total_return_pct']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        # 4. Volatility vs Returns
        ax4.scatter(df['volatility'], df['total_return_pct'], 
                   alpha=0.6, s=60, c=df['total_return_pct'], cmap='RdYlGn')
        ax4.set_xlabel('Volatility', fontsize=12)
        ax4.set_ylabel('Total Return (%)', fontsize=12)
        ax4.set_title('Risk vs Return Profile', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(visuals_dir / "manager_performance_3year.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_quarterly_activity_chart(self, visuals_dir: Path) -> None:
        """Create quarterly activity timeline visualization."""
        timeline_file = Path(self.output_dir) / "historical" / "quarterly_activity_timeline.csv"
        if not timeline_file.exists():
            return
        
        df = pd.read_csv(timeline_file)
        if df.empty:
            return
        
        # Create timeline chart
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
        fig.suptitle('Quarterly Market Activity Timeline (2007-2025)', fontsize=20, fontweight='bold')
        
        # Convert quarter to datetime for plotting
        # Parse quarters like "2025Q1" to datetime
        df['quarter_year'] = df['quarter'].str.extract(r'(\d{4})')[0].astype(int)
        df['quarter_num'] = df['quarter'].str.extract(r'Q(\d)')[0].astype(int)
        
        # Create proper date columns for pd.to_datetime
        df['month'] = (df['quarter_num'] - 1) * 3 + 1
        df['day'] = 1
        df['date'] = pd.to_datetime(df[['quarter_year', 'month', 'day']].rename(columns={'quarter_year': 'year'}))
        df = df.sort_values('date')
        
        # 1. Transaction Volume Over Time
        ax1.plot(df['date'], df['total_transactions'], marker='o', linewidth=2, markersize=4)
        ax1.fill_between(df['date'], df['total_transactions'], alpha=0.3)
        ax1.set_ylabel('Total Transactions', fontsize=12)
        ax1.set_title('Market Activity Volume', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # 2. Active Manager Count
        ax2.plot(df['date'], df['active_managers'], marker='s', linewidth=2, markersize=4, color='orange')
        ax2.fill_between(df['date'], df['active_managers'], alpha=0.3, color='orange')
        ax2.set_ylabel('Active Managers', fontsize=12)
        ax2.set_title('Manager Participation', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # 3. Average Portfolio Impact
        ax3.plot(df['date'], df['avg_portfolio_impact'], marker='^', linewidth=2, markersize=4, color='green')
        ax3.fill_between(df['date'], df['avg_portfolio_impact'], alpha=0.3, color='green')
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Avg Portfolio Impact (%)', fontsize=12)
        ax3.set_title('Impact Intensity', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Highlight major market events
        events = [
            ('2008', 'Financial Crisis'),
            ('2020', 'COVID-19'),
            ('2022', 'Rate Hikes')
        ]
        
        for event_year, event_name in events:
            event_date = pd.to_datetime(f'{event_year}-01-01')
            if event_date >= df['date'].min() and event_date <= df['date'].max():
                for ax in [ax1, ax2, ax3]:
                    ax.axvline(event_date, color='red', linestyle='--', alpha=0.7)
                    ax.text(event_date, ax.get_ylim()[1] * 0.9, event_name, 
                           rotation=90, ha='right', va='top', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(visuals_dir / "quarterly_activity_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_top_plays_chart(self, visuals_dir: Path) -> None:
        """Create top performing plays visualization."""
        plays_file = Path(self.output_dir) / "historical" / "top_performing_plays.csv"
        if not plays_file.exists():
            return
        
        df = pd.read_csv(plays_file)
        if df.empty:
            return
        
        # Filter for completed plays with returns
        completed_plays = df[df['estimated_return_pct'].notna()].copy()
        if completed_plays.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Top Performing Investment Plays Analysis', fontsize=20, fontweight='bold')
        
        # 1. Top 15 Best Returns
        top_winners = completed_plays.head(15)
        bars = ax1.barh(range(len(top_winners)), top_winners['estimated_return_pct'],
                       color=plt.cm.RdYlGn(np.linspace(0.5, 0.9, len(top_winners))))
        ax1.set_yticks(range(len(top_winners)))
        ax1.set_yticklabels([f"{row['manager_name']}: {row['ticker']}" 
                            for _, row in top_winners.iterrows()], fontsize=10)
        ax1.set_xlabel('Return (%)', fontsize=12)
        ax1.set_title('Top 15 Investment Plays', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_winners['estimated_return_pct'])):
            ax1.text(bar.get_width() + max(top_winners['estimated_return_pct']) * 0.01,
                    bar.get_y() + bar.get_height()/2, f'{value:.1f}%',
                    ha='left', va='center', fontweight='bold', fontsize=9)
        
        # 2. Holding Period vs Returns
        ax2.scatter(completed_plays['holding_days'], completed_plays['estimated_return_pct'],
                   alpha=0.6, s=50, c=completed_plays['estimated_return_pct'], cmap='RdYlGn')
        ax2.set_xlabel('Holding Days', fontsize=12)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.set_title('Holding Period vs Returns', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Annotate top performers
        for _, row in top_winners.head(5).iterrows():
            if pd.notna(row['holding_days']):
                ax2.annotate(f"{row['ticker']}", 
                           (row['holding_days'], row['estimated_return_pct']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        # 3. Manager Success Rate
        manager_stats = completed_plays.groupby('manager_name').agg({
            'estimated_return_pct': ['count', 'mean']
        }).round(1)
        manager_stats.columns = ['total_plays', 'avg_return']
        manager_stats = manager_stats[manager_stats['total_plays'] >= 3]  # Minimum 3 plays
        top_managers = manager_stats.nlargest(10, 'avg_return')
        
        bars = ax3.bar(range(len(top_managers)), top_managers['avg_return'],
                      color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_managers))))
        ax3.set_xticks(range(len(top_managers)))
        ax3.set_xticklabels(top_managers.index, rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Average Return (%)', fontsize=12)
        ax3.set_title('Top Manager Performance (Min 3 Plays)', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Portfolio Impact Distribution
        bins = [-100, -50, -10, 10, 50, 100, float('inf')]
        labels = ['<-50%', '-50 to -10%', '-10 to 10%', '10 to 50%', '50 to 100%', '>100%']
        completed_plays['return_bucket'] = pd.cut(completed_plays['estimated_return_pct'], 
                                                 bins=bins, labels=labels)
        
        bucket_counts = completed_plays['return_bucket'].value_counts()
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
        ax4.pie(bucket_counts.values, labels=bucket_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax4.set_title('Return Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(visuals_dir / "top_performing_plays.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_distribution_chart(self, visuals_dir: Path) -> None:
        """Create performance distribution and comparison charts."""
        performance_file = Path(self.output_dir) / "historical" / "manager_performance_3year_summary.csv"
        if not performance_file.exists():
            return
        
        df = pd.read_csv(performance_file)
        if df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Manager Performance Distribution Analysis', fontsize=18, fontweight='bold')
        
        # 1. Return Distribution Histogram
        ax1.hist(df['total_return_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(df['total_return_pct'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["total_return_pct"].mean():.1f}%')
        ax1.axvline(df['total_return_pct'].median(), color='green', linestyle='--',
                   label=f'Median: {df["total_return_pct"].median():.1f}%')
        ax1.set_xlabel('Total Return (%)', fontsize=12)
        ax1.set_ylabel('Number of Managers', fontsize=12)
        ax1.set_title('Return Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Transaction Activity Distribution
        ax2.scatter(df['buy_transactions'], df['sell_transactions'], 
                   c=df['total_return_pct'], cmap='RdYlGn', alpha=0.7, s=80)
        ax2.set_xlabel('Buy Transactions', fontsize=12)
        ax2.set_ylabel('Sell Transactions', fontsize=12)
        ax2.set_title('Trading Activity Patterns', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add diagonal line
        max_val = max(df['buy_transactions'].max(), df['sell_transactions'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal Buys/Sells')
        ax2.legend()
        
        # 3. Top vs Bottom Performers Comparison
        top_5 = df.head(5)
        bottom_5 = df.tail(5)
        
        categories = ['Avg Position\nImpact', 'Volatility', 'Total\nTransactions']
        top_means = [top_5['avg_position_impact'].mean(), 
                    top_5['volatility'].mean(),
                    top_5['total_transactions'].mean()]
        bottom_means = [bottom_5['avg_position_impact'].mean(),
                       bottom_5['volatility'].mean(), 
                       bottom_5['total_transactions'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, top_means, width, label='Top 5 Performers', 
                       color='lightgreen', alpha=0.8)
        bars2 = ax3.bar(x + width/2, bottom_means, width, label='Bottom 5 Performers',
                       color='lightcoral', alpha=0.8)
        
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Top vs Bottom Performer Characteristics', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Performance vs Portfolio Size
        ax4.scatter(df['end_portfolio_value'], df['total_return_pct'],
                   alpha=0.6, s=80, c=df['total_return_pct'], cmap='RdYlGn')
        ax4.set_xlabel('End Portfolio Value ($)', fontsize=12)
        ax4.set_ylabel('Total Return (%)', fontsize=12)
        ax4.set_title('Portfolio Size vs Performance', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Add annotations for outliers
        outliers = df[(df['total_return_pct'] > df['total_return_pct'].quantile(0.9)) |
                     (df['total_return_pct'] < df['total_return_pct'].quantile(0.1))]
        for _, row in outliers.iterrows():
            ax4.annotate(row['manager_name'], 
                        (row['end_portfolio_value'], row['total_return_pct']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(visuals_dir / "performance_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_historical_analysis(self) -> None:
        """Create historical trend analysis and visualizations."""
        if self.history_df is None:
            logging.warning("No history data available for historical analysis")
            return

        # Convert period to datetime
        self.history_df["period_date"] = pd.to_datetime(
            self.history_df["period"], format="%Y-%m-%d", errors="coerce"
        )

        # Filter out invalid dates
        valid_dates = self.history_df["period_date"].notna()
        if not valid_dates.any():
            logging.warning("No valid dates found in history data")
            return

        # Create quarterly activity timeline
        valid_history = self.history_df[valid_dates]
        quarterly_activity = (
            valid_history.groupby([pd.Grouper(key="period_date", freq="QE"), "action"])
            .size()
            .unstack(fill_value=0)
        )

        if len(quarterly_activity) > 0:
            # Create visualization
            fig, ax = plt.subplots(figsize=(14, 8))

            quarterly_activity.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                colormap="Set3",
            )

            ax.set_xlabel("Quarter")
            ax.set_ylabel("Number of Activities")
            ax.set_title("Quarterly Portfolio Activity Timeline")
            ax.legend(title="Action Type", bbox_to_anchor=(1.05, 1), loc="upper left")

            # Format x-axis
            ax.set_xticklabels(
                [
                    f"{d.year} Q{(d.month - 1) // 3 + 1}"
                    if hasattr(d, "strftime")
                    else str(d)
                    for d in quarterly_activity.index
                ],
                rotation=45,
                ha="right",
            )

            plt.tight_layout()
            plt.savefig(
                self.historical_visual_dir / "activity_timeline.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Save data
            quarterly_activity.to_csv(
                self.historical_dir / "quarterly_activity_timeline.csv"
            )

            logging.info("Created historical activity timeline")

    def generate_all_reports(self) -> None:
        """Generate all analysis reports and visualizations."""
        logging.info("Starting comprehensive holdings analysis...")

        # Ensure data is loaded
        if self.holdings_df is None:
            self.load_data()
        
        # Check if we have any data to analyze
        if self.holdings_df is None or self.holdings_df.empty:
            logging.warning("No holdings data available for analysis")
            return

        # Create historical analysis directory
        historical_dir = self.output_dir / "historical"
        historical_dir.mkdir(exist_ok=True)
        
        # Generate analysis reports
        analyses = {
            # Core analyses
            "interesting_stocks_overview": self.analyze_interesting_stocks_overview,
            "top_holdings": self.analyze_top_holdings,
            "high_conviction_stocks": self.analyze_high_conviction_stocks,
            "multi_manager_favorites": self.analyze_multi_manager_favorites,
            "hidden_gems": self.analyze_hidden_gems,
            
            # Performance analyses
            "manager_performance": self.analyze_manager_performance,
            "highest_portfolio_concentration": self.analyze_highest_portfolio_concentration,
            
            # Activity analyses
            "momentum_stocks": self.analyze_momentum_stocks,
            "new_positions": self.analyze_new_positions,
            "52_week_low_buys": self.analyze_52_week_low_buys,
            "52_week_high_sells": self.analyze_52_week_high_sells,
            "most_sold_stocks": self.analyze_most_sold_stocks,
            "contrarian_plays": self.analyze_contrarian_plays,
            "concentration_changes": self.analyze_concentration_changes,
            
            # Value and sector analyses
            "value_opportunities": self.analyze_value_opportunities,
            "sector_distribution": self.analyze_sector_distribution,
            "sector_rotation": self.analyze_sector_rotation,
        }
        
        # Historical analyses (saved to historical/ subdirectory)
        historical_analyses = {
            "historical_concentration_changes": self.analyze_historical_concentration_changes,
            "historical_manager_performance": self.analyze_historical_manager_performance,
            "manager_performance_3year_summary": self.analyze_manager_performance_3year,
            "quarterly_activity_timeline": self.analyze_quarterly_activity_timeline,
            "top_performing_plays": self.analyze_top_performing_plays,
            "stock_performance_winners_losers": self.analyze_historical_stock_performance,
        }
        
        # Price-based analyses
        price_analyses = [
            (5, "stocks_under_$5"),
            (10, "stocks_under_$10"),
            (20, "stocks_under_$20"),
            (50, "stocks_under_$50"),
            (100, "stocks_under_$100"),
        ]
        
        for max_price, label in price_analyses:
            analyses[label] = lambda mp=max_price, l=label: self.analyze_stocks_by_price(mp, l)
        
        # Market cap analyses
        cap_analyses = [
            ("Micro-Cap", "micro_cap_favorites"),
            ("Small-Cap", "small_cap_favorites"),
            ("Mid-Cap", "mid_cap_favorites"),
            ("Large-Cap", "large_cap_favorites"),
            ("Mega-Cap", "mega_cap_favorites"),
        ]
        
        for cap_category, label in cap_analyses:
            analyses[label] = lambda cc=cap_category: self.analyze_stocks_by_market_cap(cc)

        # Generate current analyses
        for name, analysis_func in analyses.items():
            try:
                logging.info(f"Generating {name} analysis...")
                df = analysis_func()
                if len(df) > 0:
                    output_file = self.output_dir / f"{name}.csv"
                    df.to_csv(output_file, index=False)
                    logging.info(f"Saved {name} analysis to {output_file}")
                else:
                    logging.warning(f"No data for {name} analysis")
            except Exception as e:
                logging.error(f"Error generating {name} analysis: {e}")
        
        # Generate historical analyses
        for name, analysis_func in historical_analyses.items():
            try:
                logging.info(f"Generating {name} analysis...")
                df = analysis_func()
                if len(df) > 0:
                    output_file = historical_dir / f"{name}.csv"
                    df.to_csv(output_file, index=False)
                    logging.info(f"Saved {name} analysis to {output_file}")
                else:
                    logging.warning(f"No data for {name} analysis")
            except Exception as e:
                logging.error(f"Error generating {name} analysis: {e}")

        # Generate visualizations
        visualizations = [
            self.create_manager_performance_chart,
            self.create_top_holdings_chart,
            self.create_historical_analysis,
            self.create_enhanced_normal_visualizations,
        ]

        for viz_func in visualizations:
            try:
                viz_func()
            except Exception as e:
                logging.error(f"Error creating visualization {viz_func.__name__}: {e}")

        # Generate historical visualizations
        try:
            self.create_historical_visualizations()
        except Exception as e:
            logging.error(f"Error creating historical visualizations: {e}")

        # Generate summary report
        self._generate_summary_report()

        logging.info("Analysis complete! Check the 'analysis' directory for current results and 'analysis/historical' for historical data.")

    def _generate_summary_report(self) -> None:
        """Generate a summary report of all analyses."""
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "data_summary": {
                "total_holdings": len(self.holdings_df)
                if self.holdings_df is not None
                else 0,
                "unique_stocks": len(self.holdings_df["ticker"].unique())
                if self.holdings_df is not None
                else 0,
                "total_managers": len(self.holdings_df["manager_id"].unique())
                if self.holdings_df is not None
                else 0,
                "total_value": float(self.holdings_df["value"].sum())
                if self.holdings_df is not None
                else 0,
            },
            "top_sectors": [],
            "top_holdings": [],
            "most_active_managers": [],
        }

        # Add top sectors
        sector_data = self.analyze_sector_distribution()
        if len(sector_data) > 0:
            summary["top_sectors"] = sector_data.head(5).to_dict("records")

        # Add top holdings
        top_holdings = self.analyze_top_holdings()
        if len(top_holdings) > 0:
            summary["top_holdings"] = top_holdings.head(10)[
                ["ticker", "manager_count", "total_value"]
            ].to_dict("records")

        # Add most active managers
        manager_data = self.analyze_manager_performance()
        if len(manager_data) > 0:
            summary["most_active_managers"] = manager_data.head(5)[
                ["manager_name", "position_count", "total_value"]
            ].to_dict("records")

        # Save summary
        with open(self.output_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Create README
        readme_content = f"""# Portfolio Analysis Summary

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Data Overview
- Total Holdings: {summary["data_summary"]["total_holdings"]:,}
- Unique Stocks: {summary["data_summary"]["unique_stocks"]:,}
- Total Managers: {summary["data_summary"]["total_managers"]:,}
- Total Portfolio Value: ${summary["data_summary"]["total_value"]:,.2f}

## Analysis Files Generated

### 🚀 Start Here - Best Overall Opportunities
- `interesting_stocks_overview.csv` - **BEST STARTING POINT** - Stocks scored by multiple factors
- `high_conviction_stocks.csv` - Where managers put big money (>5% positions)
- `multi_manager_favorites.csv` - Consensus picks (5+ managers)
- `value_opportunities.csv` - Cheap stocks with strong fundamentals
- `hidden_gems.csv` - Under-followed quality stocks

### 📈 Core Analysis Reports
- `top_holdings.csv` - Stocks held by multiple managers
- `sector_distribution.csv` - Portfolio distribution by sector
- `manager_performance.csv` - Manager portfolio metrics
- `momentum_stocks.csv` - Stocks with recent buying activity

### 🔄 Recent Activity & Timing
- `new_positions.csv` - Fresh buys by managers
- `52_week_low_buys.csv` - Stocks being bought at discounts
- `52_week_high_sells.csv` - Profit taking signals
- `most_sold_stocks.csv` - Stocks managers are exiting
- `contrarian_plays.csv` - Mixed buy/sell signals
- `concentration_changes.csv` - Big position changes

### 💵 Price-Based Screening
- `stocks_under_$5.csv` - Speculative/turnaround plays
- `stocks_under_$10.csv` - Small cap value hunting
- `stocks_under_$20.csv` - Overlooked opportunities
- `stocks_under_$50.csv` - Mid-range value stocks
- `stocks_under_$100.csv` - Accessible blue chips

### 📊 Market Cap Analysis
- `micro_cap_favorites.csv` - Micro-cap stocks (<$300M)
- `small_cap_favorites.csv` - Small-cap stocks ($300M-$2B)
- `mid_cap_favorites.csv` - Mid-cap stocks ($2B-$10B)
- `large_cap_favorites.csv` - Large-cap stocks ($10B-$200B)
- `mega_cap_favorites.csv` - Mega-cap stocks (>$200B)

### 🌍 Sector & Concentration
- `sector_rotation.csv` - Money flows by sector
- `highest_portfolio_concentration.csv` - Highest conviction positions

### Visualizations
- `visuals/sector_distribution.png` - Sector allocation charts
- `visuals/manager_performance.png` - Manager portfolio analysis
- `visuals/top_holdings_by_managers.png` - Most popular holdings
- `historical/visuals/activity_timeline.png` - Historical activity trends

### Summary Files
- `analysis_summary.json` - Complete analysis summary in JSON format
- `README.md` - This file

## Top 5 Holdings by Manager Count
"""

        if len(top_holdings) > 0:
            for _, holding in top_holdings.head(5).iterrows():
                readme_content += (
                    f"\n- **{holding['ticker']}**: "
                    f"{holding['manager_count']} managers, "
                    f"${holding['total_value'] / 1e9:.2f}B total value"
                )

        readme_content += "\n\n## How to Use This Data\n\n"
        readme_content += "1. Review CSV files for detailed analysis\n"
        readme_content += "2. Check visualizations for quick insights\n"
        readme_content += "3. Use analysis_summary.json for programmatic access\n"
        readme_content += "4. Historical data provides trend analysis\n"

        with open(self.output_dir / "README.md", "w") as f:
            f.write(readme_content)

    def analyze_historical_stock_performance(self) -> pd.DataFrame:
        """Analyze historical stock performance winners and losers."""
        if self.history_df is None:
            logging.warning("No history data available for stock performance analysis")
            return pd.DataFrame()
        
        # Calculate 3-year lookback from latest date
        latest_date = pd.to_datetime(self.history_df["date"]).max()
        three_years_ago = latest_date - pd.DateOffset(years=3)
        
        # Filter to 3-year period and significant activities
        three_year_data = self.history_df[
            (pd.to_datetime(self.history_df["date"]) >= three_years_ago) &
            (self.history_df["portfolio_impact"].abs() >= 0.5)  # Minimum 0.5% impact
        ].copy()
        
        if three_year_data.empty:
            logging.warning("No significant stock activity data for 3-year analysis")
            return pd.DataFrame()
        
        # Analyze performance by ticker
        stock_performance = []
        
        for ticker in three_year_data["ticker"].unique():
            ticker_data = three_year_data[three_year_data["ticker"] == ticker].copy()
            ticker_data["date"] = pd.to_datetime(ticker_data["date"])
            ticker_data = ticker_data.sort_values("date")
            
            # Calculate performance metrics
            total_activities = len(ticker_data)
            managers_holding = ticker_data["manager_id"].nunique()
            
            # Separate buys and sells
            buys = ticker_data[ticker_data["action_class"] == "buy"]
            sells = ticker_data[ticker_data["action_class"] == "sell"]
            
            # Calculate buy/sell activity
            total_buy_impact = buys["portfolio_impact"].sum()
            total_sell_impact = abs(sells["portfolio_impact"].sum())
            net_activity_score = total_buy_impact - total_sell_impact
            
            # Calculate average portfolio impact
            avg_impact = ticker_data["portfolio_impact"].abs().mean()
            
            # Get timespan
            period_start = ticker_data["date"].min()
            period_end = ticker_data["date"].max()
            days_tracked = (period_end - period_start).days
            
            # Recent activity (last 6 months)
            six_months_ago = latest_date - pd.DateOffset(months=6)
            recent_data = ticker_data[ticker_data["date"] >= six_months_ago]
            recent_activity = len(recent_data)
            recent_net_activity = 0
            if not recent_data.empty:
                recent_buys = recent_data[recent_data["action_class"] == "buy"]["portfolio_impact"].sum()
                recent_sells = abs(recent_data[recent_data["action_class"] == "sell"]["portfolio_impact"].sum())
                recent_net_activity = recent_buys - recent_sells
            
            # Classify performance
            if net_activity_score > 2.0:
                performance_category = "Strong Winner"
            elif net_activity_score > 0.5:
                performance_category = "Winner"
            elif net_activity_score > -0.5:
                performance_category = "Neutral"
            elif net_activity_score > -2.0:
                performance_category = "Loser"
            else:
                performance_category = "Strong Loser"
            
            # Get top managers for this stock
            top_managers = ticker_data.groupby("manager_id")["portfolio_impact"].sum().abs().nlargest(3)
            top_manager_names = [self.manager_name_map.get(mid, mid) for mid in top_managers.index]
            
            stock_performance.append({
                "ticker": ticker,
                "total_activities": total_activities,
                "managers_holding": managers_holding,
                "total_buy_impact": total_buy_impact,
                "total_sell_impact": total_sell_impact,
                "net_activity_score": net_activity_score,
                "avg_portfolio_impact": avg_impact,
                "days_tracked": days_tracked,
                "recent_activity_count": recent_activity,
                "recent_net_activity": recent_net_activity,
                "performance_category": performance_category,
                "period_start": period_start,
                "period_end": period_end,
                "top_3_managers": "; ".join(top_manager_names[:3])
            })
        
        performance_df = pd.DataFrame(stock_performance)
        
        if not performance_df.empty:
            # Sort by net activity score (best performers first)
            performance_df = performance_df.sort_values("net_activity_score", ascending=False)
            
            # Save to historical folder
            historical_dir = Path(self.output_dir) / "historical"
            historical_dir.mkdir(exist_ok=True)
            
            performance_df.round(2).to_csv(
                historical_dir / "stock_performance_winners_losers.csv", index=False
            )
            
            logging.info(f"Created stock performance analysis for {len(performance_df)} stocks")
        
        return performance_df

    def create_enhanced_normal_visualizations(self) -> None:
        """Create enhanced visualizations for normal analysis with more comprehensive data."""
        try:
            # Create comprehensive holdings overview chart
            self._create_comprehensive_holdings_chart()
            
            # Create enhanced manager concentration chart
            self._create_enhanced_manager_chart()
            
            # Create market dynamics chart
            self._create_market_dynamics_chart()
            
            logging.info("Created enhanced normal analysis visualizations")
            
        except Exception as e:
            logging.error(f"Error creating enhanced visualizations: {e}")

    def _create_comprehensive_holdings_chart(self) -> None:
        """Create comprehensive holdings overview with multiple perspectives."""
        if self.holdings_df is None or self.holdings_df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Comprehensive Holdings Analysis Overview', fontsize=20, fontweight='bold')
        
        # 1. Top Holdings by Value (Bubble Chart)
        ticker_summary = self.holdings_df.groupby("ticker").agg({
            "value": "sum",
            "manager_id": "nunique",
            "portfolio_percent": "mean"
        }).reset_index()
        
        top_holdings = ticker_summary.nlargest(20, "value")
        
        scatter = ax1.scatter(top_holdings["manager_id"], top_holdings["value"] / 1e9,
                            s=top_holdings["portfolio_percent"] * 20,  # Size by avg portfolio %
                            alpha=0.6, c=top_holdings["value"] / 1e9, cmap='viridis')
        
        ax1.set_xlabel('Number of Managers Holding', fontsize=12)
        ax1.set_ylabel('Total Value (Billions $)', fontsize=12)
        ax1.set_title('Top Holdings: Value vs Manager Count\n(Bubble size = Avg Portfolio %)', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add labels for top 5
        for _, row in top_holdings.head(5).iterrows():
            ax1.annotate(row['ticker'], 
                        (row['manager_id'], row['value'] / 1e9),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # 2. Manager Portfolio Size Distribution
        manager_summary = self.holdings_df.groupby("manager_id").agg({
            "value": "sum",
            "ticker": "nunique"
        }).reset_index()
        manager_summary["manager_name"] = manager_summary["manager_id"].map(self.manager_name_map)
        
        # Create size buckets
        portfolio_sizes = manager_summary["value"] / 1e9
        bins = [0, 1, 5, 10, 50, float('inf')]
        labels = ['<$1B', '$1-5B', '$5-10B', '$10-50B', '>$50B']
        manager_summary["size_bucket"] = pd.cut(portfolio_sizes, bins=bins, labels=labels)
        
        bucket_counts = manager_summary["size_bucket"].value_counts()
        colors = ['lightcoral', 'orange', 'yellow', 'lightgreen', 'darkgreen']
        wedges, texts, autotexts = ax2.pie(bucket_counts.values, labels=bucket_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Manager Portfolio Size Distribution', fontsize=14, fontweight='bold')
        
        # 3. Portfolio Concentration Analysis
        manager_summary["avg_position_size"] = manager_summary["value"] / manager_summary["ticker"]
        top_managers = manager_summary.nlargest(15, "value")
        
        bars = ax3.barh(range(len(top_managers)), top_managers["ticker"],
                       color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top_managers))))
        ax3.set_yticks(range(len(top_managers)))
        ax3.set_yticklabels([f"{row['manager_name']}" for _, row in top_managers.iterrows()], fontsize=10)
        ax3.set_xlabel('Number of Unique Stocks', fontsize=12)
        ax3.set_title('Portfolio Diversification (Top 15 Managers)', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Value Distribution Histogram
        all_position_values = self.holdings_df["value"] / 1e6  # In millions
        ax4.hist(all_position_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Position Value (Millions $)', fontsize=12)
        ax4.set_ylabel('Number of Positions', fontsize=12)
        ax4.set_title('Position Value Distribution', fontsize=14, fontweight='bold')
        ax4.axvline(all_position_values.mean(), color='red', linestyle='--',
                   label=f'Mean: ${all_position_values.mean():.1f}M')
        ax4.axvline(all_position_values.median(), color='green', linestyle='--',
                   label=f'Median: ${all_position_values.median():.1f}M')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_xlim(0, np.percentile(all_position_values, 95))  # Remove outliers for better view
        
        plt.tight_layout()
        plt.savefig(self.visual_dir / "comprehensive_holdings_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_enhanced_manager_chart(self) -> None:
        """Create enhanced manager analysis chart."""
        if self.holdings_df is None or self.holdings_df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Enhanced Manager Analysis', fontsize=20, fontweight='bold')
        
        # Get manager data
        manager_data = self.holdings_df.groupby("manager_id").agg({
            "value": ["sum", "mean", "std"],
            "portfolio_percent": ["mean", "max"],
            "ticker": "nunique"
        }).round(2)
        
        manager_data.columns = ["total_value", "mean_position", "position_volatility", 
                               "avg_portfolio_pct", "max_position_pct", "stock_count"]
        manager_data["manager_name"] = manager_data.index.map(self.manager_name_map)
        manager_data = manager_data.sort_values("total_value", ascending=False)
        
        # 1. Manager Risk-Return Profile
        top_20_managers = manager_data.head(20)
        scatter = ax1.scatter(top_20_managers["position_volatility"] / 1e6, 
                            top_20_managers["avg_portfolio_pct"],
                            s=top_20_managers["total_value"] / 1e8,  # Size by total value
                            alpha=0.6, c=top_20_managers["stock_count"], cmap='viridis')
        
        ax1.set_xlabel('Position Volatility (Millions $)', fontsize=12)
        ax1.set_ylabel('Average Portfolio Allocation (%)', fontsize=12)
        ax1.set_title('Manager Risk Profile\n(Size = Total Value, Color = Stock Count)', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Number of Stocks', fontsize=10)
        
        # 2. Concentration vs Diversification
        ax2.scatter(manager_data["stock_count"], manager_data["max_position_pct"],
                   alpha=0.6, s=60, c=manager_data["total_value"] / 1e9, cmap='plasma')
        ax2.set_xlabel('Number of Stocks', fontsize=12)
        ax2.set_ylabel('Largest Position (%)', fontsize=12)
        ax2.set_title('Portfolio Concentration Strategy', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(manager_data["stock_count"], manager_data["max_position_pct"], 1)
        p = np.poly1d(z)
        ax2.plot(manager_data["stock_count"], p(manager_data["stock_count"]), "r--", alpha=0.8)
        
        # 3. Top 15 Managers by Value
        top_15 = manager_data.head(15)
        bars = ax3.barh(range(len(top_15)), top_15["total_value"] / 1e9,
                       color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top_15))))
        ax3.set_yticks(range(len(top_15)))
        ax3.set_yticklabels([name[:30] for name in top_15["manager_name"]], fontsize=10)
        ax3.set_xlabel('Total Portfolio Value (Billions $)', fontsize=12)
        ax3.set_title('Top 15 Managers by Portfolio Value', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, value in enumerate(top_15["total_value"] / 1e9):
            ax3.text(value + max(top_15["total_value"] / 1e9) * 0.01, i,
                    f'${value:.1f}B', ha='left', va='center', fontweight='bold', fontsize=9)
        
        # 4. Position Size Distribution by Manager Type
        # Categorize managers by portfolio size
        manager_data["size_category"] = pd.cut(manager_data["total_value"] / 1e9, 
                                             bins=[0, 1, 10, 50, float('inf')],
                                             labels=['Small (<$1B)', 'Medium ($1-10B)', 
                                                   'Large ($10-50B)', 'Mega (>$50B)'])
        
        size_categories = manager_data["size_category"].value_counts()
        wedges, texts, autotexts = ax4.pie(size_categories.values, labels=size_categories.index,
                                          autopct='%1.1f%%', startangle=90,
                                          colors=['lightcoral', 'orange', 'lightgreen', 'darkgreen'])
        ax4.set_title('Manager Categories by Portfolio Size', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.visual_dir / "enhanced_manager_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_market_dynamics_chart(self) -> None:
        """Create market dynamics and trends visualization."""
        if self.holdings_df is None or self.holdings_df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Market Dynamics and Investment Trends', fontsize=20, fontweight='bold')
        
        # 1. Position Size vs Manager Interest
        ticker_analysis = self.holdings_df.groupby("ticker").agg({
            "value": ["sum", "mean"],
            "manager_id": "nunique",
            "portfolio_percent": ["mean", "max"]
        }).round(2)
        
        ticker_analysis.columns = ["total_value", "avg_position_value", "manager_count", 
                                 "avg_portfolio_pct", "max_portfolio_pct"]
        ticker_analysis = ticker_analysis.sort_values("total_value", ascending=False)
        
        # Get top 30 for visualization
        top_tickers = ticker_analysis.head(30)
        
        scatter = ax1.scatter(top_tickers["manager_count"], top_tickers["total_value"] / 1e9,
                            s=top_tickers["avg_portfolio_pct"] * 15,  # Size by avg allocation
                            alpha=0.6, c=top_tickers["max_portfolio_pct"], cmap='Reds')
        
        ax1.set_xlabel('Number of Managers Holding', fontsize=12)
        ax1.set_ylabel('Total Value (Billions $)', fontsize=12)
        ax1.set_title('Stock Popularity vs Total Investment\n(Size = Avg %, Color = Max %)', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add labels for most interesting stocks
        for _, row in top_tickers.head(8).iterrows():
            ax1.annotate(row.name, 
                        (row['manager_count'], row['total_value'] / 1e9),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        # 2. Investment Concentration Analysis
        # Calculate Herfindahl-Hirschman Index for each manager
        hhi_data = []
        for manager_id in self.holdings_df["manager_id"].unique():
            manager_holdings = self.holdings_df[self.holdings_df["manager_id"] == manager_id]
            total_value = manager_holdings["value"].sum()
            if total_value > 0:
                shares = (manager_holdings["value"] / total_value) ** 2
                hhi = shares.sum() * 10000  # Scale to 0-10000
                hhi_data.append({
                    "manager_id": manager_id,
                    "manager_name": self.manager_name_map.get(manager_id, manager_id),
                    "hhi": hhi,
                    "stock_count": len(manager_holdings),
                    "total_value": total_value
                })
        
        hhi_df = pd.DataFrame(hhi_data)
        hhi_df = hhi_df.sort_values("hhi", ascending=False)
        
        # Plot HHI distribution
        ax2.scatter(hhi_df["stock_count"], hhi_df["hhi"],
                   alpha=0.6, s=60, c=hhi_df["total_value"] / 1e9, cmap='viridis')
        ax2.set_xlabel('Number of Stocks', fontsize=12)
        ax2.set_ylabel('Concentration Index (HHI)', fontsize=12)
        ax2.set_title('Portfolio Concentration Index', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add concentration levels
        ax2.axhline(2500, color='red', linestyle='--', alpha=0.7, label='Highly Concentrated')
        ax2.axhline(1500, color='orange', linestyle='--', alpha=0.7, label='Moderately Concentrated')
        ax2.legend()
        
        # 3. Top Conviction Plays (Highest Portfolio Allocations)
        conviction_plays = self.holdings_df.nlargest(20, "portfolio_percent")
        
        bars = ax3.barh(range(len(conviction_plays)), conviction_plays["portfolio_percent"],
                       color=plt.cm.Reds(np.linspace(0.4, 0.9, len(conviction_plays))))
        ax3.set_yticks(range(len(conviction_plays)))
        ax3.set_yticklabels([f"{row['ticker']} ({self.manager_name_map.get(row['manager_id'], row['manager_id'])[:15]})" 
                            for _, row in conviction_plays.iterrows()], fontsize=9)
        ax3.set_xlabel('Portfolio Allocation (%)', fontsize=12)
        ax3.set_title('Top 20 Conviction Plays', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Market Cap Distribution (if available)
        if hasattr(self, 'stocks_df') and self.stocks_df is not None and not self.stocks_df.empty:
            # Merge with market cap data
            holdings_with_mcap = self.holdings_df.merge(
                self.stocks_df[["ticker", "market_cap"]], on="ticker", how="left"
            )
            holdings_with_mcap = holdings_with_mcap[holdings_with_mcap["market_cap"].notna()]
            
            if not holdings_with_mcap.empty:
                # Create market cap buckets
                mcap_billions = holdings_with_mcap["market_cap"] / 1e9
                bins = [0, 2, 10, 50, 200, float('inf')]
                labels = ['Small Cap\n(<$2B)', 'Mid Cap\n($2-10B)', 'Large Cap\n($10-50B)', 
                         'Mega Cap\n($50-200B)', 'Giant Cap\n(>$200B)']
                holdings_with_mcap["mcap_bucket"] = pd.cut(mcap_billions, bins=bins, labels=labels)
                
                # Calculate total investment by market cap bucket
                mcap_investment = holdings_with_mcap.groupby("mcap_bucket")["value"].sum()
                
                wedges, texts, autotexts = ax4.pie(mcap_investment.values, labels=mcap_investment.index,
                                                  autopct='%1.1f%%', startangle=90)
                ax4.set_title('Investment Allocation by Market Cap', fontsize=14, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Market Cap Data\nNot Available', 
                        ha='center', va='center', fontsize=16, transform=ax4.transAxes)
                ax4.set_title('Market Cap Analysis', fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Market Cap Data\nNot Available', 
                    ha='center', va='center', fontsize=16, transform=ax4.transAxes)
            ax4.set_title('Market Cap Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.visual_dir / "market_dynamics_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main() -> None:
    """Main entry point for the analysis script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze portfolio holdings from Dataroma"
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory containing cached data files",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory for output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create analyzer and run analysis
    analyzer = HoldingsAnalyzer(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    try:
        analyzer.load_data()
        analyzer.generate_all_reports()
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
