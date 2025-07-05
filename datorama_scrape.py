#!/usr/bin/env python3
"""
Optimized Dataroma scraper for extracting investment portfolio data.

This module scrapes manager holdings and activities from dataroma.com
with intelligent caching and fallback mechanisms.
"""

import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

# Progress Update:
# ✓ Set up imports with proper ordering and type hints
# → Working on core scraper class implementation

# Ensure log directory exists
Path("log").mkdir(exist_ok=True)


def is_wsl1_environment() -> bool:
    """Detect if running in WSL1 which has poor GUI support."""
    try:
        if platform.system() != "Linux":
            return False

        # Check for WSL in kernel version
        with open("/proc/version") as f:
            version = f.read().lower()
            if "microsoft" not in version:
                return False

        # Check if it's WSL1 by looking for WSL version
        try:
            result = subprocess.run(
                ["wsl.exe", "--status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if (
                "version 1" in result.stdout.lower()
                or "version: 1" in result.stdout.lower()
            ):
                return True
        except Exception:
            pass

        # WSL1 detection based on kernel version patterns
        # WSL1 typically has 4.4.x kernels and specific build patterns
        if "microsoft" in version and "4.4" in version:
            return True

        # Additional WSL1 patterns (older GCC, specific build patterns)
        # But exclude WSL2 patterns
        if "WSL2" in version or "standard-WSL" in version or "5.15" in version:
            return False

        return "microsoft" in version and any(
            pattern in version for pattern in ["-microsoft", "gcc version 5.4"]
        )
    except Exception:
        return False


# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("log/dataroma_scraper.log"),
        logging.StreamHandler(),
    ],
)


class OptimizedDataromaScraper:
    """Optimized scraper for Dataroma investment data."""

    def __init__(
        self, use_playwright_for_stocks: bool = True, headless: bool = True
    ) -> None:
        """
        Initialize the optimized scraper.

        Args:
            use_playwright_for_stocks: Whether to use JS rendering for stock pages only
            headless: Whether to run browser in headless mode
        """
        self.base_url = "https://www.dataroma.com/m/"
        self.cache_dir = "cache"
        self.html_dir = "cache/html"  # Same as original

        # Detect WSL1 and disable Playwright if necessary
        if use_playwright_for_stocks and is_wsl1_environment():
            logging.warning(
                "⚠️  WSL1 detected - Playwright may hang. "
                "Disabling browser fallback for stability."
            )
            logging.info(
                "💡 Using yfinance as primary backup instead of browser automation."
            )
            self.use_playwright_for_stocks = False
        else:
            self.use_playwright_for_stocks = use_playwright_for_stocks

        self.headless = headless
        self.ensure_directories()

        # Progress tracking
        self.progress = {
            "managers_processed": 0,
            "holdings_found": 0,
            "activities_found": 0,
            "stocks_processed": 0,
            "stocks_with_market_cap": 0,
            "stocks_with_pe": 0,
        }

        # Enhanced browser headers
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Session with retry
        self.session = self._create_session()

        # Cache management
        self.cache_duration = timedelta(days=1)  # Cache HTML for 1 day
        self.stock_cache_duration = timedelta(days=7)  # Cache stock data for 7 days
        self.load_caches()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        session.headers.update(self.headers)
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.cache_dir,
            self.html_dir,
            f"{self.html_dir}/general",
            f"{self.html_dir}/managers",
            f"{self.html_dir}/stocks",
            f"{self.cache_dir}/stocks",
            "analysis",
            "analysis/visuals",
            "analysis/historical",
            "analysis/historical/visuals",
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_caches(self) -> None:
        """Load existing cache files."""
        self.cache_files = {
            "managers": f"{self.cache_dir}/managers.json",
            "holdings": f"{self.cache_dir}/holdings.json",
            "holdings_by_manager": f"{self.cache_dir}/holdings_by_manager.json",
            "holdings_by_ticker": f"{self.cache_dir}/holdings_by_ticker.json",
            "holdings_enriched": f"{self.cache_dir}/holdings_enriched.json",
            "history": f"{self.cache_dir}/history.json",
            "history_by_manager": f"{self.cache_dir}/history_by_manager.json",
            "history_by_ticker": f"{self.cache_dir}/history_by_ticker.json",
            "stocks": f"{self.cache_dir}/stocks.json",
            "overview": f"{self.cache_dir}/overview.json",
            "metadata": f"{self.cache_dir}/metadata.json",
            "enrichment_history": f"{self.cache_dir}/enrichment_history.json",
            "data_quality_fixes": f"{self.cache_dir}/data_quality_fixes.json",
        }

        # Load existing data
        self.cached_data = {}
        for key, filepath in self.cache_files.items():
            try:
                with open(filepath) as f:
                    self.cached_data[key] = json.load(f)
                    logging.info(
                        f"✓ Loaded {key} cache: {len(self.cached_data[key])} items"
                    )
            except FileNotFoundError:
                self.cached_data[key] = {} if key != "history" else []
            except json.JSONDecodeError:
                logging.warning(f"⚠️  Corrupted cache file: {filepath}, creating new")
                self.cached_data[key] = {} if key != "history" else []

    def save_cache(self, cache_type: str) -> None:
        """Save cache data to file."""
        if cache_type in self.cache_files:
            try:
                with open(self.cache_files[cache_type], "w") as f:
                    json.dump(self.cached_data[cache_type], f, indent=2)
                logging.debug(f"✓ Saved {cache_type} cache")
            except Exception as e:
                logging.error(f"❌ Failed to save {cache_type} cache: {e}")

    def save_all_caches(self) -> None:
        """Save all cache data."""
        for cache_type in self.cache_files:
            self.save_cache(cache_type)

    def get_cached_html(self, url: str, cache_key: str) -> Optional[str]:  # noqa: ARG002
        """
        Get cached HTML content if still valid.

        Args:
            url: URL to check cache for
            cache_key: Key to use for caching

        Returns:
            Cached HTML content or None if not cached/expired
        """
        cache_path = Path(f"{self.html_dir}/{cache_key}")

        if not cache_path.exists():
            return None

        # Check if cache is still valid
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_age > self.cache_duration:
            logging.debug(f"Cache expired for {cache_key}")
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                logging.debug(f"Using cached HTML for {cache_key}")
                return f.read()
        except Exception as e:
            logging.error(f"Error reading cache {cache_key}: {e}")
            return None

    def save_html_cache(self, html: str, cache_key: str) -> None:
        """Save HTML content to cache."""
        cache_path = Path(f"{self.html_dir}/{cache_key}")
        try:
            # Ensure parent directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(html)
            logging.debug(f"Saved HTML cache for {cache_key}")
        except Exception as e:
            logging.error(f"Error saving cache {cache_key}: {e}")

    def fetch_page(self, url: str, cache_key: Optional[str] = None) -> Optional[str]:
        """
        Fetch a page with caching support.

        Args:
            url: URL to fetch
            cache_key: Optional cache key for storing HTML

        Returns:
            HTML content or None if failed
        """
        # Check cache first
        if cache_key:
            cached_html = self.get_cached_html(url, cache_key)
            if cached_html:
                return cached_html

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html = response.text

            # Save to cache
            if cache_key:
                self.save_html_cache(html, cache_key)

            return html
        except Exception as e:
            logging.error(f"❌ Error fetching {url}: {e}")
            return None

    async def fetch_page_with_js(
        self, url: str, cache_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Fetch a page using Playwright for JavaScript rendering.

        Args:
            url: URL to fetch
            cache_key: Optional cache key for storing HTML

        Returns:
            HTML content or None if failed
        """
        # Check cache first
        if cache_key:
            cached_html = self.get_cached_html(url, cache_key)
            if cached_html:
                return cached_html

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                page = await browser.new_page()

                # Set viewport and user agent
                await page.set_viewport_size({"width": 1920, "height": 1080})
                await page.set_extra_http_headers(self.headers)

                # Navigate with timeout
                await page.goto(url, wait_until="networkidle", timeout=60000)

                # Wait for content
                await page.wait_for_timeout(2000)

                # Get HTML
                html = await page.content()

                await browser.close()

                # Save to cache
                if cache_key and html:
                    self.save_html_cache(html, cache_key)

                return html

        except PlaywrightTimeoutError:
            logging.error(f"⏱️  Timeout loading {url}")
            return None
        except Exception as e:
            logging.error(f"❌ Playwright error for {url}: {e}")
            return None

    def parse_managers_list(self, html: str) -> list[dict[str, Any]]:
        """
        Parse the managers list from HTML.

        Args:
            html: HTML content of managers page

        Returns:
            List of manager dictionaries
        """
        soup = BeautifulSoup(html, "html.parser")
        managers = []

        # Find all manager links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # Manager pages follow pattern: /m/managers.php?m=XX
            if "/m/managers.php?m=" in href:
                manager_id = href.split("m=")[-1]
                manager_name = link.text.strip()

                if manager_id and manager_name:
                    managers.append(
                        {
                            "id": manager_id,
                            "name": manager_name,
                            "url": f"{self.base_url}managers.php?m={manager_id}",
                        }
                    )

        return managers

    def get_managers_list(self) -> list[dict[str, Any]]:
        """Get list of all managers."""
        logging.info("📋 Fetching managers list...")

        cache_key = "general/managers_page.html"
        url = f"{self.base_url}home.php"

        html = self.fetch_page(url, cache_key)
        if not html:
            logging.error("❌ Failed to fetch managers list")
            return []

        managers = self.parse_managers_list(html)

        # Update cache
        self.cached_data["managers"] = {m["id"]: m for m in managers}
        self.save_cache("managers")

        logging.info(f"✓ Found {len(managers)} managers")
        return managers

    def parse_holdings(self, html: str, manager_id: str) -> list[dict[str, Any]]:
        """
        Parse holdings from manager page HTML.

        Args:
            html: HTML content of manager page
            manager_id: Manager identifier

        Returns:
            List of holding dictionaries
        """
        soup = BeautifulSoup(html, "html.parser")
        holdings = []

        # Look for the holdings table
        holdings_table = soup.find("table", {"id": "grid"})
        if not holdings_table:
            logging.warning(f"No holdings table found for {manager_id}")
            return holdings

        # Get the current period from the page
        period_elem = soup.find("span", {"class": "period"})
        period = period_elem.text.strip() if period_elem else "Unknown"

        # Parse each row
        for row in holdings_table.find_all("tr")[1:]:  # Skip header
            cells = row.find_all("td")
            if len(cells) >= 7:
                try:
                    # Extract stock ticker from link
                    stock_link = cells[0].find("a")
                    if not stock_link:
                        continue

                    ticker_match = re.search(
                        r"stock\.php\?ticker=([^&]+)", stock_link["href"]
                    )
                    if not ticker_match:
                        continue

                    ticker = ticker_match.group(1)

                    holding = {
                        "manager_id": manager_id,
                        "ticker": ticker,
                        "stock": cells[0].text.strip(),
                        "portfolio_percentage": self._parse_percentage(cells[1].text),
                        "shares": self._parse_number(cells[2].text),
                        "recent_activity": cells[3].text.strip(),
                        "reported_price": self._parse_currency(cells[4].text),
                        "value": self._parse_currency(cells[5].text),
                        "period": period,
                        "timestamp": datetime.now().isoformat(),
                    }

                    holdings.append(holding)

                except Exception as e:
                    logging.error(f"Error parsing holding row: {e}")
                    continue

        return holdings

    def get_manager_holdings(self, manager: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Get holdings for a specific manager.

        Args:
            manager: Manager dictionary with id, name, url

        Returns:
            List of holding dictionaries
        """
        manager_id = manager["id"]
        logging.info(f"📊 Fetching holdings for {manager['name']} ({manager_id})...")

        # Check if we have recent data
        if manager_id in self.cached_data.get("holdings_by_manager", {}):
            cached = self.cached_data["holdings_by_manager"][manager_id]
            if cached.get("timestamp"):
                cache_age = datetime.now() - datetime.fromisoformat(cached["timestamp"])
                if cache_age < self.cache_duration:
                    logging.debug(f"Using cached holdings for {manager_id}")
                    return cached["holdings"]

        cache_key = f"managers/{manager_id}/holdings_{int(time.time())}.html"
        html = self.fetch_page(manager["url"], cache_key)

        if not html:
            logging.error(f"❌ Failed to fetch holdings for {manager_id}")
            return []

        holdings = self.parse_holdings(html, manager_id)

        # Update holdings cache
        if manager_id not in self.cached_data["holdings_by_manager"]:
            self.cached_data["holdings_by_manager"][manager_id] = {}

        self.cached_data["holdings_by_manager"][manager_id] = {
            "holdings": holdings,
            "timestamp": datetime.now().isoformat(),
            "manager_name": manager["name"],
        }

        # Update holdings by ticker cache
        for holding in holdings:
            ticker = holding["ticker"]
            if ticker not in self.cached_data["holdings_by_ticker"]:
                self.cached_data["holdings_by_ticker"][ticker] = []

            # Add this holding to ticker's list
            self.cached_data["holdings_by_ticker"][ticker].append(
                {
                    "manager_id": manager_id,
                    "manager_name": manager["name"],
                    "portfolio_percentage": holding["portfolio_percentage"],
                    "shares": holding["shares"],
                    "value": holding["value"],
                    "period": holding["period"],
                }
            )

        self.save_cache("holdings_by_manager")
        self.save_cache("holdings_by_ticker")

        self.progress["holdings_found"] += len(holdings)
        logging.info(f"✓ Found {len(holdings)} holdings for {manager['name']}")

        return holdings

    def parse_activity(self, html: str, manager_id: str) -> list[dict[str, Any]]:
        """
        Parse activity history from manager activity page.

        Args:
            html: HTML content of activity page
            manager_id: Manager identifier

        Returns:
            List of activity dictionaries
        """
        soup = BeautifulSoup(html, "html.parser")
        activities = []

        # Look for the activity table
        activity_table = soup.find("table", {"id": "grid"})
        if not activity_table:
            logging.warning(f"No activity table found for {manager_id}")
            return activities

        # Parse each row
        for row in activity_table.find_all("tr")[1:]:  # Skip header
            cells = row.find_all("td")
            if len(cells) >= 7:
                try:
                    # Extract stock ticker from link
                    stock_link = cells[1].find("a")
                    if not stock_link:
                        continue

                    ticker_match = re.search(
                        r"stock\.php\?ticker=([^&]+)", stock_link["href"]
                    )
                    if not ticker_match:
                        continue

                    ticker = ticker_match.group(1)

                    activity = {
                        "manager_id": manager_id,
                        "period": cells[0].text.strip(),
                        "ticker": ticker,
                        "stock": cells[1].text.strip(),
                        "action": cells[2].text.strip(),
                        "shares": self._parse_number(cells[3].text),
                        "portfolio_percentage": self._parse_percentage(cells[4].text),
                        "purchase_price": self._parse_currency(cells[5].text),
                        "current_price": self._parse_currency(cells[6].text),
                        "timestamp": datetime.now().isoformat(),
                    }

                    activities.append(activity)

                except Exception as e:
                    logging.error(f"Error parsing activity row: {e}")
                    continue

        return activities

    async def get_manager_activity_async(
        self, manager: dict[str, Any], page: int = 1
    ) -> list[dict[str, Any]]:
        """
        Get activity history for a manager using async Playwright.

        Args:
            manager: Manager dictionary
            page: Page number to fetch

        Returns:
            List of activity dictionaries
        """
        manager_id = manager["id"]
        url = f"{self.base_url}activity.php?m={manager_id}&p={page}"
        cache_key = f"managers/{manager_id}/activity_page{page}_{int(time.time())}.html"

        html = await self.fetch_page_with_js(url, cache_key)
        if not html:
            logging.error(f"❌ Failed to fetch activity page {page} for {manager_id}")
            return []

        return self.parse_activity(html, manager_id)

    def get_manager_activity(
        self, manager: dict[str, Any], max_pages: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get all activity history for a manager.

        Args:
            manager: Manager dictionary
            max_pages: Maximum number of pages to fetch

        Returns:
            List of activity dictionaries
        """
        manager_id = manager["id"]
        logging.info(f"📈 Fetching activity for {manager['name']} ({manager_id})...")

        all_activities = []

        # Check cache first
        if manager_id in self.cached_data.get("history_by_manager", {}):
            cached = self.cached_data["history_by_manager"][manager_id]
            if cached.get("timestamp"):
                cache_age = datetime.now() - datetime.fromisoformat(cached["timestamp"])
                if cache_age < self.cache_duration:
                    logging.debug(f"Using cached activity for {manager_id}")
                    return cached["activities"]

        # Fetch first page synchronously to check if we need Playwright
        url = f"{self.base_url}activity.php?m={manager_id}&p=1"
        cache_key = f"managers/{manager_id}/activity_page1_{int(time.time())}.html"
        html = self.fetch_page(url, cache_key)

        if html:
            activities = self.parse_activity(html, manager_id)
            all_activities.extend(activities)

            # Check if there are more pages
            soup = BeautifulSoup(html, "html.parser")
            page_links = soup.find_all(
                "a", href=re.compile(r"activity\.php\?m=.*&p=\d+")
            )
            total_pages = 1
            for link in page_links:
                match = re.search(r"p=(\d+)", link["href"])
                if match:
                    total_pages = max(total_pages, int(match.group(1)))

            # Fetch remaining pages if needed
            if total_pages > 1 and self.use_playwright_for_stocks:
                pages_to_fetch = min(total_pages, max_pages)
                logging.info(
                    f"Found {total_pages} activity pages, "
                    f"fetching up to {pages_to_fetch}"
                )

                # Use async to fetch remaining pages
                async def fetch_all_pages():
                    tasks = []
                    for page_num in range(2, pages_to_fetch + 1):
                        task = self.get_manager_activity_async(manager, page_num)
                        tasks.append(task)

                    return await asyncio.gather(*tasks)

                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # If loop is already running (e.g., in Jupyter), create task
                    additional_activities = []
                else:
                    additional_activities = loop.run_until_complete(fetch_all_pages())

                for activities in additional_activities:
                    all_activities.extend(activities)

        # Update cache
        if manager_id not in self.cached_data["history_by_manager"]:
            self.cached_data["history_by_manager"][manager_id] = {}

        self.cached_data["history_by_manager"][manager_id] = {
            "activities": all_activities,
            "timestamp": datetime.now().isoformat(),
            "manager_name": manager["name"],
        }

        # Update history by ticker
        for activity in all_activities:
            ticker = activity["ticker"]
            if ticker not in self.cached_data["history_by_ticker"]:
                self.cached_data["history_by_ticker"][ticker] = []

            self.cached_data["history_by_ticker"][ticker].append(
                {
                    "manager_id": manager_id,
                    "manager_name": manager["name"],
                    "period": activity["period"],
                    "action": activity["action"],
                    "shares": activity["shares"],
                    "portfolio_percentage": activity["portfolio_percentage"],
                }
            )

        self.save_cache("history_by_manager")
        self.save_cache("history_by_ticker")

        self.progress["activities_found"] += len(all_activities)
        logging.info(f"✓ Found {len(all_activities)} activities for {manager['name']}")

        return all_activities

    def get_stock_data_yfinance(self, ticker: str) -> dict[str, Any]:
        """
        Get stock data using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract relevant metrics
            data = {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "eps": info.get("trailingEps"),
                "revenue": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "currency": info.get("currency", "USD"),
                "country": info.get("country"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary", "")[
                    :500
                ],  # Limit description
                "employees": info.get("fullTimeEmployees"),
                "data_source": "yfinance",
                "last_updated": datetime.now().isoformat(),
            }

            # Clean up None values
            return {k: v for k, v in data.items() if v is not None}

        except Exception as e:
            logging.error(f"❌ yfinance error for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "data_source": "yfinance_error",
                "last_updated": datetime.now().isoformat(),
            }

    def parse_stock_page(self, html: str, ticker: str) -> dict[str, Any]:
        """
        Parse stock data from Dataroma stock page.

        Args:
            html: HTML content of stock page
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock metrics
        """
        soup = BeautifulSoup(html, "html.parser")
        data = {"ticker": ticker, "data_source": "dataroma"}

        try:
            # Try to find market data
            market_data = soup.find("div", {"class": "market_data"})
            if market_data:
                # Extract market cap
                market_cap_elem = market_data.find(text=re.compile(r"Market Cap"))
                if market_cap_elem:
                    market_cap_text = market_cap_elem.find_next().text
                    data["market_cap"] = self._parse_market_cap(market_cap_text)

                # Extract P/E ratio
                pe_elem = market_data.find(text=re.compile(r"P/E"))
                if pe_elem:
                    pe_text = pe_elem.find_next().text
                    data["pe_ratio"] = self._parse_number(pe_text)

            # Try to extract from any table data
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        label = cells[0].text.strip().lower()
                        value = cells[1].text.strip()

                        if "market cap" in label:
                            data["market_cap"] = self._parse_market_cap(value)
                        elif "p/e" in label or "pe ratio" in label:
                            data["pe_ratio"] = self._parse_number(value)
                        elif "price" in label and "current" in label:
                            data["price"] = self._parse_currency(value)

        except Exception as e:
            logging.error(f"Error parsing stock page for {ticker}: {e}")

        data["last_updated"] = datetime.now().isoformat()
        return data

    async def get_stock_data_browser(self, ticker: str) -> dict[str, Any]:
        """
        Get stock data using browser rendering.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock metrics
        """
        url = f"{self.base_url}stock.php?ticker={ticker}"
        cache_key = f"stocks/{ticker}_{int(time.time())}.html"

        html = await self.fetch_page_with_js(url, cache_key)
        if not html:
            logging.error(f"❌ Failed to fetch stock page for {ticker}")
            return {
                "ticker": ticker,
                "error": "Failed to fetch",
                "data_source": "browser_error",
            }

        return self.parse_stock_page(html, ticker)

    def get_stock_data(self, ticker: str) -> dict[str, Any]:
        """
        Get stock data with fallback mechanisms.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock metrics
        """
        # Check cache first
        cache_file = f"{self.cache_dir}/stocks/{ticker}.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached_data = json.load(f)
                    if "last_updated" in cached_data:
                        cache_age = datetime.now() - datetime.fromisoformat(
                            cached_data["last_updated"]
                        )
                        if cache_age < self.stock_cache_duration:
                            logging.debug(f"Using cached data for {ticker}")
                            return cached_data
            except Exception as e:
                logging.warning(f"Error reading cache for {ticker}: {e}")

        # Try yfinance first
        logging.info(f"📊 Fetching data for {ticker}...")
        data = self.get_stock_data_yfinance(ticker)

        # If yfinance failed and we have browser support, try that
        if (
            data.get("error")
            and self.use_playwright_for_stocks
            and not is_wsl1_environment()
        ):
            logging.info(f"Trying browser fallback for {ticker}...")
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # Running in async context
                browser_data = {
                    "ticker": ticker,
                    "error": "Cannot use browser in async context",
                }
            else:
                browser_data = loop.run_until_complete(
                    self.get_stock_data_browser(ticker)
                )

            # Merge data, preferring non-error values
            if not browser_data.get("error"):
                data.update(browser_data)

        # Save to cache
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving cache for {ticker}: {e}")

        # Update progress
        if data.get("market_cap"):
            self.progress["stocks_with_market_cap"] += 1
        if data.get("pe_ratio"):
            self.progress["stocks_with_pe"] += 1

        return data

    def enrich_holdings(self, holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Enrich holdings with additional stock data.

        Args:
            holdings: List of holding dictionaries

        Returns:
            Enriched holdings list
        """
        logging.info(f"🔧 Enriching {len(holdings)} holdings with stock data...")

        # Get unique tickers
        unique_tickers = list({h["ticker"] for h in holdings})
        logging.info(f"Found {len(unique_tickers)} unique tickers")

        # Get stock data for each ticker
        stock_data_map = {}
        for i, ticker in enumerate(unique_tickers, 1):
            logging.info(f"Processing ticker {i}/{len(unique_tickers)}: {ticker}")
            stock_data = self.get_stock_data(ticker)
            stock_data_map[ticker] = stock_data
            self.progress["stocks_processed"] += 1

            # Save progress periodically
            if i % 10 == 0:
                self.save_all_caches()

        # Enrich holdings
        enriched_holdings = []
        for holding in holdings:
            enriched = holding.copy()
            ticker = holding["ticker"]

            if ticker in stock_data_map:
                stock_info = stock_data_map[ticker]
                enriched.update(
                    {
                        "market_cap": stock_info.get("market_cap"),
                        "pe_ratio": stock_info.get("pe_ratio"),
                        "forward_pe": stock_info.get("forward_pe"),
                        "current_price": stock_info.get("price"),
                        "52_week_high": stock_info.get("52_week_high"),
                        "52_week_low": stock_info.get("52_week_low"),
                        "dividend_yield": stock_info.get("dividend_yield"),
                        "sector": stock_info.get("sector"),
                        "industry": stock_info.get("industry"),
                        "country": stock_info.get("country"),
                        "exchange": stock_info.get("exchange"),
                        "data_quality": "complete"
                        if stock_info.get("market_cap") and stock_info.get("pe_ratio")
                        else "partial",
                    }
                )

            enriched_holdings.append(enriched)

        # Save enriched data
        self.cached_data["holdings_enriched"] = enriched_holdings
        self.cached_data["stocks"] = stock_data_map
        self.save_cache("holdings_enriched")
        self.save_cache("stocks")

        logging.info(
            f"✓ Enrichment complete. {self.progress['stocks_with_market_cap']} "
            f"stocks with market cap, {self.progress['stocks_with_pe']} with P/E ratio"
        )

        return enriched_holdings

    def scrape_all(
        self, limit_managers: Optional[int] = None, skip_enrichment: bool = False
    ) -> dict[str, Any]:
        """
        Main method to scrape all data.

        Args:
            limit_managers: Optional limit on number of managers to process
            skip_enrichment: Whether to skip stock data enrichment

        Returns:
            Dictionary containing all scraped data
        """
        start_time = datetime.now()
        logging.info("🚀 Starting optimized Dataroma scrape...")

        # Get managers list
        managers = self.get_managers_list()
        if limit_managers:
            managers = managers[:limit_managers]
            logging.info(f"Limited to {limit_managers} managers")

        # Collect all holdings and activities
        all_holdings = []
        all_activities = []

        for i, manager in enumerate(managers, 1):
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Processing manager {i}/{len(managers)}: {manager['name']}")

            # Get holdings
            holdings = self.get_manager_holdings(manager)
            all_holdings.extend(holdings)

            # Get activity history
            activities = self.get_manager_activity(manager)
            all_activities.extend(activities)

            self.progress["managers_processed"] += 1

            # Save progress
            if i % 5 == 0:
                self.save_all_caches()

        # Enrich with stock data if requested
        if not skip_enrichment and all_holdings:
            all_holdings = self.enrich_holdings(all_holdings)

        # Create overview
        unique_stocks = len({h["ticker"] for h in all_holdings})
        overview = {
            "last_updated": datetime.now().isoformat(),
            "managers_count": len(managers),
            "total_holdings": len(all_holdings),
            "unique_stocks": unique_stocks,
            "total_activities": len(all_activities),
            "data_quality": {
                "stocks_with_market_cap": self.progress["stocks_with_market_cap"],
                "stocks_with_pe": self.progress["stocks_with_pe"],
                "enrichment_rate": (
                    self.progress["stocks_with_market_cap"] / unique_stocks * 100
                    if unique_stocks > 0
                    else 0
                ),
            },
            "scrape_duration": str(datetime.now() - start_time),
        }

        # Save final data
        self.cached_data["overview"] = overview
        self.cached_data["holdings"] = all_holdings
        self.cached_data["history"] = all_activities
        self.save_all_caches()

        # Save last update time
        with open(f"{self.cache_dir}/last_update.json", "w") as f:
            json.dump({"timestamp": datetime.now().isoformat()}, f)

        # Print summary
        self._print_summary(overview)

        return {
            "overview": overview,
            "managers": managers,
            "holdings": all_holdings,
            "activities": all_activities,
            "progress": self.progress,
        }

    def _print_summary(self, overview: dict[str, Any]) -> None:
        """Print scraping summary."""
        print("\n" + "=" * 60)
        print("📊 SCRAPING COMPLETE!")
        print("=" * 60)
        print(f"✓ Managers processed: {overview['managers_count']}")
        print(f"✓ Total holdings: {overview['total_holdings']}")
        print(f"✓ Unique stocks: {overview['unique_stocks']}")
        print(f"✓ Total activities: {overview['total_activities']}")
        print(
            f"✓ Market cap data: "
            f"{overview['data_quality']['stocks_with_market_cap']} stocks"
        )
        print(f"✓ P/E ratio data: {overview['data_quality']['stocks_with_pe']} stocks")
        print(f"✓ Enrichment rate: {overview['data_quality']['enrichment_rate']:.1f}%")
        print(f"✓ Duration: {overview['scrape_duration']}")
        print("=" * 60)

    # Utility methods
    def _parse_percentage(self, text: str) -> Optional[float]:
        """Parse percentage from text."""
        try:
            return float(text.strip().rstrip("%"))
        except (ValueError, AttributeError):
            return None

    def _parse_number(self, text: str) -> Optional[int]:
        """Parse number from text, handling commas."""
        try:
            return int(text.strip().replace(",", ""))
        except (ValueError, AttributeError):
            return None

    def _parse_currency(self, text: str) -> Optional[float]:
        """Parse currency value from text."""
        try:
            return float(text.strip().lstrip("$").replace(",", ""))
        except (ValueError, AttributeError):
            return None

    def _parse_market_cap(self, text: str) -> Optional[float]:
        """Parse market cap from text like $1.5B or $500M."""
        try:
            text = text.strip().upper()
            # Remove dollar sign and spaces
            text = text.replace("$", "").replace(" ", "").replace(",", "")

            # Extract number and multiplier
            multipliers = {"B": 1e9, "BILLION": 1e9, "M": 1e6, "MILLION": 1e6, "K": 1e3}

            for suffix, multiplier in multipliers.items():
                if text.endswith(suffix):
                    number = float(text[: -len(suffix)])
                    return number * multiplier

            # Try to parse as regular number
            return float(text)
        except (ValueError, AttributeError):
            return None


def main() -> None:
    """Main entry point for the scraper."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Dataroma portfolio scraper")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of managers to scrape",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable browser rendering for stock pages",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run browser in headful mode (visible)",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip stock data enrichment",
    )
    parser.add_argument(
        "--managers",
        nargs="+",
        help="Specific manager IDs to scrape",
    )

    args = parser.parse_args()

    # Initialize scraper
    scraper = OptimizedDataromaScraper(
        use_playwright_for_stocks=not args.no_browser,
        headless=not args.headful,
    )

    # Run scraping
    if args.managers:
        # Scrape specific managers
        managers = [
            {
                "id": m_id,
                "name": f"Manager {m_id}",
                "url": f"{scraper.base_url}managers.php?m={m_id}",
            }
            for m_id in args.managers
        ]
        all_holdings = []
        all_activities = []

        for manager in managers:
            holdings = scraper.get_manager_holdings(manager)
            all_holdings.extend(holdings)
            activities = scraper.get_manager_activity(manager)
            all_activities.extend(activities)

        if not args.skip_enrichment:
            all_holdings = scraper.enrich_holdings(all_holdings)

        scraper.save_all_caches()
    else:
        # Full scrape
        scraper.scrape_all(
            limit_managers=args.limit,
            skip_enrichment=args.skip_enrichment,
        )


if __name__ == "__main__":
    main()
