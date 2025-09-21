# Data Models Module/
# │
# ├── Data Classes/
# │   ├── ETFInfo # ETF metadata model (ticker, name, aum, description, timestamps)
# │   ├── ETFDataPoint # Single ETF data point model (OHLCV + RSI data for one date)
# │   └── ETFAnalysis # ETF analysis results model (current metrics, signals, categories)
# │
# └── ETFDataModel/
#     │
#     ├── Data Transformation/
#     │   ├── from_yfinance_data(ticker, yf_data) # Convert yfinance DataFrame to standardized format
#     │   └── filter_by_period(df, period) # Filter data by time period (1mo, 3mo, 6mo, 1y, 2y, 5y)
#     │
#     ├── Signal Analysis/
#     │   ├── calculate_signals(df) # Generate trading signals based on RSI thresholds
#     │   └── get_latest_analysis(df, ticker) # Extract latest analysis as ETFAnalysis object
#     │
#     └── Data Quality/
#         └── validate_data_quality(df) # Comprehensive data validation with quality metrics

# Data Class Details:
# ├── ETFInfo/
# │   ├── ticker: str # ETF ticker symbol (required)
# │   ├── name: Optional[str] # ETF full name
# │   ├── aum: Optional[int] # Assets Under Management
# │   ├── description: Optional[str] # ETF description
# │   ├── created_at: Optional[datetime] # Record creation timestamp
# │   └── updated_at: Optional[datetime] # Last update timestamp
# │
# ├── ETFDataPoint/
# │   ├── date: datetime # Trading date
# │   ├── open: float # Opening price
# │   ├── high: float # High price
# │   ├── low: float # Low price
# │   ├── close: float # Closing price
# │   ├── volume: int # Trading volume
# │   ├── adj_close: float # Adjusted closing price
# │   ├── rsi: Optional[float] # Relative Strength Index
# │   └── rsi_vol: Optional[float] # RSI volume indicator
# │
# └── ETFAnalysis/
#     ├── ticker: str # ETF ticker symbol
#     ├── current_price: float # Latest closing price
#     ├── current_rsi: float # Latest RSI value
#     ├── current_volume: int # Latest trading volume
#     ├── signal: str # Trading signal ('BUY', 'SELL', 'HOLD', 'WEAK_BUY', 'WEAK_SELL')
#     ├── signal_strength: str # Signal strength ('STRONG', 'MODERATE', 'WEAK')
#     ├── rsi_category: str # RSI category ('OVERSOLD', 'OVERBOUGHT', 'NORMAL')
#     └── last_updated: datetime # Analysis timestamp

# Signal Logic:
# ├── RSI Thresholds/
# │   ├── RSI < 30 → BUY signal (OVERSOLD)
# │   ├── RSI 30-40 → WEAK_BUY signal
# │   ├── RSI 40-60 → HOLD signal (NORMAL)
# │   ├── RSI 60-70 → WEAK_SELL signal
# │   └── RSI > 70 → SELL signal (OVERBOUGHT)
# │
# └── Signal Strength/
#     ├── RSI < 20 or > 80 → STRONG
#     ├── RSI < 35 or > 65 → MODERATE
#     └── All others → WEAK

# Data Quality Checks:
# ├── Structure Validation/
# │   ├── Required columns presence # Checks for date, OHLCV columns
# │   ├── Missing values detection # Identifies null/NaN values
# │   └── Data type consistency # Ensures proper data types
# │
# ├── Business Logic Validation/
# │   ├── Price consistency # High >= Low, High >= Open/Close, Low <= Open/Close
# │   ├── Negative value detection # Flags negative prices/volumes
# │   └── Date continuity # Checks for missing business days
# │
# └── Quality Metrics/
#     ├── is_valid: bool # Overall data validity flag
#     ├── row_count: int # Total number of records
#     ├── missing_values: dict # Count of missing values per column
#     ├── date_range: dict # Start and end dates of dataset
#     └── issues: list # List of identified data quality issues



# database/models.py - Data Models
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np
import pandas as pd

@dataclass
class ETFInfo:
    """ETF metadata model"""
    ticker: str
    name: Optional[str] = None
    aum: Optional[int] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class ETFDataPoint:
    """Single ETF data point model"""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float
    rsi: Optional[float] = None
    rsi_vol: Optional[float] = None

@dataclass
class ETFAnalysis:
    """ETF analysis results model"""
    ticker: str
    current_price: float
    current_rsi: float
    current_volume: int
    signal: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: str  # 'STRONG', 'WEAK', 'NEUTRAL'
    rsi_category: str  # 'OVERSOLD', 'OVERBOUGHT', 'NORMAL'
    last_updated: datetime

class ETFDataModel:
    """Model for ETF data operations"""
    
    @staticmethod
    def from_yfinance_data(ticker: str, yf_data: pd.DataFrame) -> pd.DataFrame:
        """Convert yfinance data to our standard format"""
        if yf_data.empty:
            return pd.DataFrame()
        
        # Standardize column names
        df = yf_data.copy()
        df.columns = df.columns.str.lower()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Add adj_close if not present
        if 'adj close' in df.columns:
            df['adj_close'] = df['adj close']
            df.drop('adj close', axis=1, inplace=True)
        elif 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
        
        # Reset index to make date a column
        if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on RSI"""
        if df.empty or 'rsi' not in df.columns:
            return df
        
        df = df.copy()
        
        # RSI-based signals
        conditions = [
            (df['rsi'] < 30),
            (df['rsi'] < 40) & (df['rsi'] >= 30),
            (df['rsi'] > 70),
            (df['rsi'] > 60) & (df['rsi'] <= 70),
        ]
        
        choices = ['BUY', 'WEAK_BUY', 'SELL', 'WEAK_SELL']
        df['signal'] = np.select(conditions, choices, default='HOLD')
        
        # Signal strength
        strength_conditions = [
            (df['rsi'] < 20) | (df['rsi'] > 80),
            (df['rsi'] < 35) | (df['rsi'] > 65),
        ]
        
        strength_choices = ['STRONG', 'MODERATE']
        df['signal_strength'] = np.select(strength_conditions, strength_choices, default='WEAK')
        
        # RSI category
        category_conditions = [
            (df['rsi'] < 30),
            (df['rsi'] > 70),
        ]
        
        category_choices = ['OVERSOLD', 'OVERBOUGHT']
        df['rsi_category'] = np.select(category_conditions, category_choices, default='NORMAL')
        
        return df
    
    @staticmethod
    def get_latest_analysis(df: pd.DataFrame, ticker: str) -> Optional[ETFAnalysis]:
        """Get latest analysis from DataFrame"""
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        
        return ETFAnalysis(
            ticker=ticker,
            current_price=float(latest.get('close', 0)),
            current_rsi=float(latest.get('rsi', 50)),
            current_volume=int(latest.get('volume', 0)),
            signal=latest.get('signal', 'HOLD'),
            signal_strength=latest.get('signal_strength', 'WEAK'),
            rsi_category=latest.get('rsi_category', 'NORMAL'),
            last_updated=pd.to_datetime(latest.get('date', datetime.now()))
        )
    
    @staticmethod
    def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter data by period (1mo, 3mo, 6mo, 1y, 2y, 5y)"""
        if df.empty:
            return df
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'])
        
        # Calculate start date based on period
        end_date = datetime.now()
        
        period_map = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        
        days = period_map.get(period, 180)  # Default to 6 months
        start_date = end_date - pd.Timedelta(days=days)
        
        # Filter data
        filtered_df = df[df['date'] >= start_date].copy()
        
        return filtered_df
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> dict[str, any]:
        """Validate data quality and return metrics"""
        if df.empty:
            return {
                'is_valid': False,
                'row_count': 0,
                'missing_values': {},
                'date_range': None,
                'issues': ['Empty dataset']
            }
        
        issues = []
        
        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for missing values
        missing_values = df.isnull().sum().to_dict()
        if any(missing_values.values()):
            issues.append("Dataset contains missing values")
        
        # Check date range
        date_range = None
        if 'date' in df.columns:
            min_date = df['date'].min()
            max_date = df['date'].max()
            date_range = {'start': min_date, 'end': max_date}
            
            # Check for gaps in dates (business days)
            expected_dates = pd.bdate_range(start=min_date, end=max_date)
            actual_dates = set(pd.to_datetime(df['date']).dt.date)
            expected_dates_set = set(expected_dates.date)
            missing_dates = expected_dates_set - actual_dates
            
            if len(missing_dates) > 0:
                issues.append(f"Missing {len(missing_dates)} business days in date range")
        
        # Check for negative values where they shouldn't be
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for logical consistency (high >= low, etc.)
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            if (df['high'] < df['low']).any():
                issues.append("High prices lower than low prices detected")
            if (df['high'] < df['close']).any() or (df['high'] < df['open']).any():
                issues.append("High prices inconsistent with open/close prices")
            if (df['low'] > df['close']).any() or (df['low'] > df['open']).any():
                issues.append("Low prices inconsistent with open/close prices")
        
        return {
            'is_valid': len(issues) == 0,
            'row_count': len(df),
            'missing_values': missing_values,
            'date_range': date_range,
            'issues': issues
        }