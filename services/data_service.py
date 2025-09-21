# ```
# DataService/
# ├── __init__(request_timeout, rate_limit_delay, last_request_time) # Initialize service with rate limiting config
# │
# ├── Rate Limiting & Control/
# │   └── _rate_limit() # Apply rate limiting between API requests to avoid throttling
# │
# ├── Core Data Fetching/
# │   ├── fetch_etf_data(ticker, period, start_date, end_date) # Fetch OHLCV data from Yahoo Finance
# │   ├── fetch_multiple_etfs(tickers, period) # Fetch data for multiple ETFs sequentially
# │   └── batch_fetch_with_retry(tickers, period, max_retries) # Batch fetch with retry logic and error handling
# │
# ├── ETF Metadata/
# │   ├── fetch_etf_info(ticker) # Fetch ETF metadata (name, AUM, description, expense ratio, yield)
# │   └── validate_ticker(ticker) # Check if ticker exists and has available data
# │
# ├── Data Gap Analysis/
# │   ├── get_missing_date_ranges(ticker, existing_dates, start_date, end_date) # Find missing date ranges
# │   └── get_period_start_date(period) # Convert period string to datetime start date
# │
# ├── Market Information/
# │   ├── fetch_market_status() # Get current market status (open, closed, pre/after market)
# │   └── get_data_freshness(ticker) # Check how recent the available data is
# │
# └── Utility Methods/
#     └── Period mapping # Convert period strings (1mo, 3mo, 6mo, 1y, 2y, 5y) to days

# Core Data Fetching Details:
# ├── fetch_etf_data()/
# │   ├── Input: ticker, period/date_range # ETF symbol and time period or specific dates
# │   ├── Process: yfinance API call → clean data → standardize columns
# │   ├── Output: DataFrame with OHLCV data # Date, Open, High, Low, Close, Volume, Adj_Close
# │   └── Features: Rate limiting, error handling, data cleaning
# │
# ├── fetch_multiple_etfs()/
# │   ├── Input: List[tickers], period # Multiple ETF symbols and time period
# │   ├── Process: Sequential fetching with individual error handling
# │   ├── Output: Dict[ticker, DataFrame] # Dictionary mapping tickers to their data
# │   └── Features: Continues on individual failures, logs warnings
# │
# └── batch_fetch_with_retry()/
#     ├── Input: List[tickers], period, max_retries # ETFs with retry configuration
#     ├── Process: Retry logic with exponential backoff
#     ├── Output: Dict[ticker, DataFrame] + failed_tickers list
#     └── Features: Configurable retries, failure tracking, progressive delays

# ETF Metadata Structure:
# ├── fetch_etf_info() returns/
# │   ├── ticker: str # ETF symbol
# │   ├── name: str # Long/short name
# │   ├── aum: int # Total assets under management
# │   ├── description: str # Business summary
# │   ├── category: str # ETF category
# │   ├── family: str # Fund family
# │   ├── expense_ratio: float # Annual expense ratio
# │   ├── yield: float # Current yield
# │   └── nav: float # Net Asset Value
# │
# └── validate_ticker()/
#     ├── Input: ticker symbol
#     ├── Process: Attempt 5-day data fetch
#     ├── Output: boolean validity
#     └── Purpose: Pre-validate tickers before bulk operations

# Data Gap Analysis:
# ├── get_missing_date_ranges()/
# │   ├── Input: ticker, existing_dates, start_date, end_date
# │   ├── Process: Generate business days → find gaps → group consecutive ranges
# │   ├── Output: List[Tuple[start_date, end_date]] # Missing date ranges
# │   └── Purpose: Optimize API calls by fetching only missing data
# │
# └── get_period_start_date()/
#     ├── Input: period string ('1mo', '3mo', '6mo', '1y', '2y', '5y')
#     ├── Process: Map to days and calculate start date
#     ├── Output: datetime object
#     └── Purpose: Convert user-friendly periods to API parameters

# Market Status Logic:
# ├── fetch_market_status()/
# │   ├── Data Source: SPY ticker as market proxy
# │   ├── Status Types: 'open', 'closed', 'pre_market', 'after_market', 'weekend', 'unknown'
# │   ├── Logic: Compare current time with trading hours (9:30 AM - 4:00 PM EST)
# │   └── Output: Dict with status, last_trading_day, last_update
# │
# └── get_data_freshness()/
#     ├── Input: ticker symbol
#     ├── Process: Fetch last 5 days → calculate age
#     ├── Freshness Rule: <= 3 days old (accounts for weekends/holidays)
#     └── Output: Dict with is_fresh, last_date, days_old

# Rate Limiting Features:
# ├── Default delay: 0.1 seconds between requests
# ├── Tracks last_request_time for precise timing
# ├── Automatic sleep calculation to maintain minimum intervals
# └── Applied to all external API calls

# Error Handling:
# ├── Individual ticker failures don't stop batch operations
# ├── Comprehensive logging at different levels (info, warning, error)
# ├── Graceful degradation with empty DataFrame returns
# ├── Retry logic with progressive delays (1s, 2s intervals)
# └── Failed ticker tracking and reporting

# Period Mapping:
# ├── '1mo' → 30 days
# ├── '3mo' → 90 days  
# ├── '6mo' → 180 days (default)
# ├── '1y' → 365 days
# ├── '2y' → 730 days
# └── '5y' → 1825 days

# Module Exports:
# ├── DataService # Main external data fetching service
# └── ETFService # ETF-specific service (imported but not defined in this file)
# ```

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# services/data_service.py - External Data Fetching Service
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
from config import Config
import requests
import time

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching external financial data"""
    
    def __init__(self):
        self.request_timeout = Config.REQUEST_TIMEOUT
        self.rate_limit_delay = 0.1  # Delay between requests to avoid rate limiting
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_etf_data(self, ticker: str, period: str = '1y', 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch ETF data from Yahoo Finance"""
        try:
            self._rate_limit()
            
            # Create yfinance Ticker object
            etf = yf.Ticker(ticker)
            
            # Fetch data
            if start_date and end_date:
                data = etf.history(start=start_date, end=end_date, interval='1d')
            else:
                data = etf.history(period=period, interval='1d')
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Clean and prepare data
            data = data.reset_index()
            data.columns = data.columns.str.lower()
            
            # Rename columns to match our schema
            column_mapping = {
                'adj close': 'adj_close'
            }
            data.rename(columns=column_mapping, inplace=True)
            
            # Add ticker column
            data['ticker'] = ticker
            
            logger.info(f"Fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_etfs(self, tickers: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple ETFs"""
        results = {}
        
        for ticker in tickers:
            try:
                data = self.fetch_etf_data(ticker, period)
                if not data.empty:
                    results[ticker] = data
                else:
                    logger.warning(f"No data available for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                continue
        
        return results
    
    def fetch_etf_info(self, ticker: str) -> dict[str, any]:
        """Fetch ETF metadata from Yahoo Finance"""
        try:
            self._rate_limit()
            
            etf = yf.Ticker(ticker)
            info = etf.info
            
            # Extract relevant information
            etf_info = {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'aum': info.get('totalAssets', 0),
                'description': info.get('longBusinessSummary', ''),
                'category': info.get('category', ''),
                'family': info.get('family', ''),
                'expense_ratio': info.get('annualReportExpenseRatio', 0),
                'yield': info.get('yield', 0),
                'nav': info.get('navPrice', 0)
            }
            
            return etf_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {'ticker': ticker, 'name': ticker}
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker exists and has data"""
        try:
            self._rate_limit()
            
            etf = yf.Ticker(ticker)
            
            # Try to fetch just 5 days of data
            data = etf.history(period='5d')
            
            return not data.empty
            
        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}")
            return False
    
    def get_missing_date_ranges(self, ticker: str, existing_dates: List[datetime],
                              start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """Find date ranges that need to be fetched"""
        if not existing_dates:
            return [(start_date, end_date)]
        
        # Convert to date objects for comparison
        existing_dates = sorted([d.date() if isinstance(d, datetime) else d for d in existing_dates])
        start_date = start_date.date() if isinstance(start_date, datetime) else start_date
        end_date = end_date.date() if isinstance(end_date, datetime) else end_date
        
        # Generate all business days in the target range
        all_business_days = pd.bdate_range(start=start_date, end=end_date)
        target_dates = set(all_business_days.date)
        existing_dates_set = set(existing_dates)
        
        # Find missing dates
        missing_dates = sorted(target_dates - existing_dates_set)
        
        if not missing_dates:
            return []
        
        # Group consecutive missing dates into ranges
        ranges = []
        current_start = missing_dates[0]
        current_end = missing_dates[0]
        
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - current_end).days <= 1:
                current_end = missing_dates[i]
            else:
                ranges.append((datetime.combine(current_start, datetime.min.time()),
                             datetime.combine(current_end, datetime.min.time())))
                current_start = missing_dates[i]
                current_end = missing_dates[i]
        
        # Add the last range
        ranges.append((datetime.combine(current_start, datetime.min.time()),
                     datetime.combine(current_end, datetime.min.time())))
        
        return ranges
    
    def fetch_market_status(self) -> dict[str, any]:
        """Get current market status"""
        try:
            # Use SPY as a proxy for market status
            spy = yf.Ticker('SPY')
            
            # Get the most recent trading day
            recent_data = spy.history(period='2d')
            
            if recent_data.empty:
                return {'status': 'unknown', 'last_update': None}
            
            last_trading_day = recent_data.index[-1]
            now = datetime.now()
            
            # Simple market status logic
            if last_trading_day.date() == now.date():
                if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                    status = 'pre_market'
                elif now.hour >= 16:
                    status = 'after_market'
                else:
                    status = 'open'
            else:
                # Check if it's a weekend
                if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    status = 'weekend'
                else:
                    status = 'closed'
            
            return {
                'status': status,
                'last_trading_day': last_trading_day,
                'last_update': now
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {'status': 'unknown', 'last_update': None}
    
    def get_period_start_date(self, period: str) -> datetime:
        """Convert period string to start date"""
        now = datetime.now()
        
        period_map = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        
        days = period_map.get(period, 180)
        return now - timedelta(days=days)
    
    def batch_fetch_with_retry(self, tickers: List[str], period: str = '1y',
                              max_retries: int = 3) -> dict[str, pd.DataFrame]:
        """Fetch multiple tickers with retry logic"""
        results = {}
        failed_tickers = []
        
        for ticker in tickers:
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    data = self.fetch_etf_data(ticker, period)
                    if not data.empty:
                        results[ticker] = data
                        success = True
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(1)  # Wait before retry
                            
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Attempt {retry_count} failed for {ticker}: {e}")
                    if retry_count < max_retries:
                        time.sleep(2)  # Wait longer on exception
            
            if not success:
                failed_tickers.append(ticker)
                logger.error(f"Failed to fetch {ticker} after {max_retries} attempts")
        
        if failed_tickers:
            logger.warning(f"Failed to fetch data for: {failed_tickers}")
        
        return results
    
    def get_data_freshness(self, ticker: str) -> dict[str, any]:
        """Check how fresh the available data is for a ticker"""
        try:
            # Get just the last few days to check freshness
            data = self.fetch_etf_data(ticker, period='5d')
            
            if data.empty:
                return {'is_fresh': False, 'last_date': None, 'days_old': None}
            
            last_date = pd.to_datetime(data['date']).max()
            now = datetime.now()
            days_old = (now - last_date).days
            
            # Consider data fresh if it's from the last trading day
            # Account for weekends and holidays
            is_fresh = days_old <= 3  # Allow up to 3 days for weekends/holidays
            
            return {
                'is_fresh': is_fresh,
                'last_date': last_date,
                'days_old': days_old
            }
            
        except Exception as e:
            logger.error(f"Error checking data freshness for {ticker}: {e}")
            return {'is_fresh': False, 'last_date': None, 'days_old': None}

# # services/__init__.py
# from .data_service import DataService
# from .etf_service import ETFService

# __all__ = ['DataService', 'ETFService']