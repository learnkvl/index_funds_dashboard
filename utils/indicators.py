# ```
# TechnicalIndicators/
# │
# ├── Momentum Oscillators/
# │   ├── compute_rsi(data, column, period=14) # Relative Strength Index with exponential smoothing
# │   ├── compute_stochastic(high, low, close, k_period=14, d_period=3) # %K and %D oscillator
# │   ├── compute_williams_r(high, low, close, period=14) # Williams %R momentum indicator
# │   ├── compute_roc(data, period=12, column) # Rate of Change percentage
# │   └── compute_momentum(data, period=10, column) # Simple momentum calculation
# │
# ├── Moving Averages/
# │   ├── compute_sma(data, period, column) # Simple Moving Average
# │   ├── compute_ema(data, period, column) # Exponential Moving Average
# │   └── compute_macd(data, fast=12, slow=26, signal=9, column) # MACD system (line, signal, histogram)
# │
# ├── Volatility Indicators/
# │   ├── compute_bollinger_bands(data, period=20, std_dev=2, column) # Upper, middle, lower bands
# │   └── compute_atr(high, low, close, period=14) # Average True Range volatility measure
# │
# ├── Volume Analysis/
# │   └── compute_volume_sma(volume, period=20) # Volume Simple Moving Average
# │
# ├── Support & Resistance/
# │   └── detect_support_resistance(data, window=20, column) # Local min/max level detection
# │
# └── Data Validation/
#     └── validate_ohlc_data(open, high, low, close) # OHLC logical consistency validation

# Momentum Oscillators Details:
# ├── compute_rsi()/
# │   ├── Input: Price data (Series/DataFrame), column name (optional), period (default 14)
# │   ├── Algorithm: Exponential smoothing of gains/losses → RS ratio → RSI formula
# │   ├── Formula: RSI = 100 - (100 / (1 + RS)), where RS = Avg Gains / Avg Losses
# │   ├── Range: 0-100 scale with overbought (>70) and oversold (<30) thresholds
# │   ├── Error Handling: Returns neutral RSI (50) for insufficient data or errors
# │   ├── Edge Cases: NaN filling, infinite value replacement, minimum data validation
# │   └── Output: pd.Series with RSI values indexed to input data
# │
# ├── compute_stochastic()/
# │   ├── Input: High, low, close price series + periods for %K and %D
# │   ├── %K Calculation: %K = 100 * ((Close - LowestLow) / (HighestHigh - LowestLow))
# │   ├── %D Calculation: %D = SMA of %K over d_period
# │   ├── Range: 0-100 scale, similar interpretation to RSI
# │   ├── Error Handling: Division by zero protection, neutral value (50) for NaN
# │   └── Output: Dict with 'k_percent' and 'd_percent' Series
# │
# ├── compute_williams_r()/
# │   ├── Input: High, low, close price series + calculation period
# │   ├── Formula: %R = -100 * ((HighestHigh - Close) / (HighestHigh - LowestLow))
# │   ├── Range: -100 to 0, where -80 to -100 = oversold, -20 to 0 = overbought
# │   ├── Error Handling: Division by zero protection, neutral value (-50) for errors
# │   └── Output: pd.Series with Williams %R values
# │
# ├── compute_roc()/
# │   ├── Input: Price data + period for rate calculation
# │   ├── Formula: ROC = ((Current Price / Price N periods ago) - 1) * 100
# │   ├── Purpose: Measures percentage change over specified time period
# │   ├── Range: Unbounded percentage values (positive = appreciation, negative = depreciation)
# │   └── Output: pd.Series with ROC percentage values
# │
# └── compute_momentum()/
#     ├── Input: Price data + period for momentum calculation
#     ├── Formula: Momentum = Current Price - Price N periods ago
#     ├── Purpose: Measures absolute price change over time
#     ├── Range: Unbounded absolute values in price units
#     └── Output: pd.Series with momentum values

# Moving Averages System:
# ├── compute_sma()/
# │   ├── Input: Price data + period for averaging
# │   ├── Algorithm: Rolling window mean calculation
# │   ├── Parameters: min_periods=1 for partial calculations
# │   ├── Purpose: Trend identification and smoothing
# │   └── Output: pd.Series with simple moving average values
# │
# ├── compute_ema()/
# │   ├── Input: Price data + span period for exponential weighting
# │   ├── Algorithm: Exponential weighted moving average (more weight to recent prices)
# │   ├── Parameters: adjust=False for consistent calculation
# │   ├── Purpose: Responsive trend following with reduced lag
# │   └── Output: pd.Series with exponential moving average values
# │
# └── compute_macd()/
#     ├── Input: Price data + fast period (12) + slow period (26) + signal period (9)
#     ├── MACD Line: EMA(fast) - EMA(slow) = trend momentum indicator
#     ├── Signal Line: EMA of MACD line over signal period = entry/exit signals
#     ├── Histogram: MACD - Signal = momentum strength indicator
#     ├── Signals: MACD crossover signal line = buy/sell opportunities
#     └── Output: Dict with 'macd', 'signal', 'histogram' Series

# Volatility Analysis:
# ├── compute_bollinger_bands()/
# │   ├── Input: Price data + period (20) + standard deviation multiplier (2)
# │   ├── Middle Band: Simple Moving Average of price
# │   ├── Upper Band: Middle Band + (StdDev * multiplier)
# │   ├── Lower Band: Middle Band - (StdDev * multiplier)
# │   ├── Purpose: Volatility measurement and overbought/oversold identification
# │   ├── Signals: Price touching bands = potential reversal points
# │   └── Output: Dict with 'upper', 'middle', 'lower' band Series
# │
# └── compute_atr()/
#     ├── Input: High, low, close price series + period (14)
#     ├── True Range Components: max(H-L, |H-Cp|, |L-Cp|) where Cp = previous close
#     ├── ATR Calculation: Exponential moving average of True Range
#     ├── Purpose: Volatility measurement for position sizing and stop-loss placement
#     ├── Range: Positive values in price units (higher = more volatile)
#     └── Output: pd.Series with Average True Range values

# Volume Analysis:
# ├── compute_volume_sma()/
# │   ├── Input: Volume Series + period for averaging
# │   ├── Purpose: Identify unusual volume activity vs historical average
# │   ├── Signals: Volume above average = stronger price moves
# │   ├── Applications: Confirm price breakouts, divergence analysis
# │   └── Output: pd.Series with volume moving average

# Support & Resistance Detection:
# ├── detect_support_resistance()/
# │   ├── Input: Price data + window size (20) for local extrema detection
# │   ├── Algorithm: Scan for local minima (support) and maxima (resistance)
# │   ├── Support Detection: Price is lowest within window = potential support level
# │   ├── Resistance Detection: Price is highest within window = potential resistance level
# │   ├── Window Logic: Look back/forward 'window' periods for extrema validation
# │   ├── Output: Dict with 'support' and 'resistance' lists of (date, price) tuples
# │   └── Applications: Key level identification, breakout analysis, trade entry/exit

# Data Input Flexibility:
# ├── Series/DataFrame Support/
# │   ├── Series Input: Direct calculation on price series
# │   ├── DataFrame Input: Requires column specification for target data
# │   ├── Column Validation: Checks for column existence in DataFrame
# │   ├── Error Messages: Clear feedback for missing columns or invalid input
# │   └── Index Preservation: Maintains original date/time indexing
# │
# └── Data Type Handling/
#     ├── Automatic Type Detection: Series vs DataFrame input handling
#     ├── Index Management: Preserves and returns consistent indexing
#     ├── Length Validation: Ensures sufficient data for calculations
#     └── Error Recovery: Graceful fallback to neutral/NaN values

# OHLC Data Validation:
# ├── validate_ohlc_data()/
# │   ├── Length Consistency: All OHLC series must have same length
# │   ├── Logical Relationships: High >= Open/Close/Low, Low <= Open/Close/High
# │   ├── Value Validation: No negative prices allowed
# │   ├── Comprehensive Checks: Validates all price relationships
# │   ├── Error Reporting: Detailed logging of validation failures
# │   └── Return Value: Boolean indicating data validity

# Error Handling Strategy:
# ├── Input Validation/
# │   ├── Data Type Checking: Series vs DataFrame validation
# │   ├── Column Existence: DataFrame column validation
# │   ├── Minimum Data Requirements: Period-specific minimum data checks
# │   └── Parameter Validation: Period and threshold value validation
# │
# ├── Calculation Protection/
# │   ├── Division by Zero: Replace zero denominators with NaN
# │   ├── Infinite Values: Replace inf/-inf with neutral values
# │   ├── NaN Handling: Fill missing values with appropriate defaults
# │   └── Edge Case Management: Handle empty data and boundary conditions
# │
# ├── Graceful Degradation/
# │   ├── Partial Calculations: Return partial results when possible
# │   ├── Neutral Values: Use neutral indicator values (RSI=50, Williams %R=-50)
# │   ├── Empty Series: Return appropriately sized empty series on total failure
# │   └── Logging: Comprehensive error logging for debugging
# │
# └── Recovery Mechanisms/
#     ├── Try-Catch Blocks: Wrap all calculations in exception handling
#     ├── Fallback Values: Provide sensible defaults for failed calculations
#     ├── Index Consistency: Maintain input indexing even in error cases
#     └── User Feedback: Clear error messages without breaking execution

# Mathematical Implementation:
# ├── RSI Algorithm: Uses exponential smoothing instead of simple averages for smoother results
# ├── MACD System: Standard 12/26/9 configuration with proper EMA calculations
# ├── Bollinger Bands: 20-period SMA with 2 standard deviation bands
# ├── Stochastic: Fast %K with smoothed %D for momentum analysis
# ├── ATR: True Range with exponential smoothing for volatility measurement
# └── Support/Resistance: Local extrema detection with configurable window sizes

# Performance Considerations:
# ├── Vectorized Operations: Uses pandas vectorized operations for speed
# ├── Memory Efficiency: Avoids unnecessary data copying
# ├── Rolling Windows: Efficient rolling calculations with min_periods
# ├── EMA Calculations: Uses pandas ewm() for optimized exponential averaging
# └── Index Preservation: Maintains original indexing for data alignment

# Output Specifications:
# ├── Series Outputs: All single-value indicators return pd.Series
# ├── Dictionary Outputs: Multi-component indicators (MACD, Bollinger) return dicts
# ├── Index Alignment: Output indices match input data indices
# ├── Data Types: Appropriate numeric types (float64) for calculations
# └── Naming Conventions: Clear, descriptive keys for dictionary outputs
# ```

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# utils/indicators.py - Technical Indicators
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    @staticmethod
    def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.Series:
        """
        Compute Supertrend indicator
        Args:
            df: DataFrame with 'high', 'low', 'close'
            period: ATR period
            multiplier: ATR multiplier
        Returns:
            pd.Series: Supertrend values
        """
        try:
            atr = TechnicalIndicators.compute_atr(df['high'], df['low'], df['close'], period)
            hl2 = (df['high'] + df['low']) / 2
            upperband = hl2 + (multiplier * atr)
            lowerband = hl2 - (multiplier * atr)
            supertrend = pd.Series(index=df.index, dtype='float64')
            direction = True  # True for uptrend, False for downtrend
            for i in range(len(df)):
                if i == 0:
                    supertrend.iloc[i] = upperband.iloc[i]
                    direction = True
                else:
                    if df['close'].iloc[i] > upperband.iloc[i-1]:
                        direction = True
                    elif df['close'].iloc[i] < lowerband.iloc[i-1]:
                        direction = False
                    if direction:
                        supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i-1])
                    else:
                        supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i-1])
            return supertrend
        except Exception as e:
            logger.error(f"Error computing Supertrend: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute Average Directional Index (ADX)
        Args:
            df: DataFrame with 'high', 'low', 'close'
            period: ADX period
        Returns:
            pd.Series: ADX values
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            plus_dm = high.diff()
            minus_dm = low.diff().abs()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean()
            plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window=period, min_periods=1).mean()
            return adx
        except Exception as e:
            logger.error(f"Error computing ADX: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    @staticmethod
    def compute_parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
        """
        Compute Parabolic SAR
        Args:
            df: DataFrame with 'high', 'low'
            step: Acceleration factor
            max_step: Maximum acceleration factor
        Returns:
            pd.Series: Parabolic SAR values
        """
        try:
            high = df['high']
            low = df['low']
            sar = pd.Series(index=df.index, dtype='float64')
            uptrend = True
            af = step
            ep = low.iloc[0]
            sar.iloc[0] = low.iloc[0]
            for i in range(1, len(df)):
                prev_sar = sar.iloc[i-1]
                if uptrend:
                    sar.iloc[i] = prev_sar + af * (ep - prev_sar)
                    if low.iloc[i] < sar.iloc[i]:
                        uptrend = False
                        sar.iloc[i] = high.iloc[i]
                        af = step
                        ep = high.iloc[i]
                    else:
                        if high.iloc[i] > ep:
                            ep = high.iloc[i]
                            af = min(af + step, max_step)
                else:
                    sar.iloc[i] = prev_sar + af * (ep - prev_sar)
                    if high.iloc[i] > sar.iloc[i]:
                        uptrend = True
                        sar.iloc[i] = low.iloc[i]
                        af = step
                        ep = low.iloc[i]
                    else:
                        if low.iloc[i] < ep:
                            ep = low.iloc[i]
                            af = min(af + step, max_step)
            return sar
        except Exception as e:
            logger.error(f"Error computing Parabolic SAR: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

    @staticmethod
    def compute_ichimoku(df: pd.DataFrame) -> dict:
        """
        Compute Ichimoku Cloud components
        Args:
            df: DataFrame with 'high', 'low', 'close'
        Returns:
            dict: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
            kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
            senkou_a = ((tenkan + kijun) / 2).shift(26)
            senkou_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
            chikou = close.shift(-26)
            return {
                'tenkan_sen': tenkan,
                'kijun_sen': kijun,
                'senkou_span_a': senkou_a,
                'senkou_span_b': senkou_b,
                'chikou_span': chikou
            }
        except Exception as e:
            logger.error(f"Error computing Ichimoku Cloud: {e}")
            empty = pd.Series([np.nan] * len(df), index=df.index)
            return {
                'tenkan_sen': empty,
                'kijun_sen': empty,
                'senkou_span_a': empty,
                'senkou_span_b': empty,
                'chikou_span': empty
            }

    @staticmethod
    def compute_linear_regression(data: Union[pd.Series, pd.DataFrame], column: str = None, period: int = 14) -> pd.Series:
        """
        Compute Linear Regression trend line
        Args:
            data: Price data (Series or DataFrame)
            column: Column name if DataFrame
            period: Regression window
        Returns:
            pd.Series: Regression line values
        """
        try:
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            reg_line = prices.rolling(window=period, min_periods=1).apply(
                lambda x: np.polyval(np.polyfit(range(len(x)), x, 1), len(x)-1), raw=True
            )
            return reg_line
        except Exception as e:
            logger.error(f"Error computing Linear Regression: {e}")
            index = data.index if hasattr(data, 'index') else range(len(data))
            return pd.Series([np.nan] * len(data), index=index)
    """Technical indicators for financial analysis"""
    
    @staticmethod
    def compute_rsi(data: Union[pd.Series, pd.DataFrame], column: str = None, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI)
        
        Args:
            data: Price data (Series or DataFrame)
            column: Column name if DataFrame is provided
            period: RSI period (default 14)
            
        Returns:
            pd.Series: RSI values
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            # Ensure we have enough data
            if len(prices) < period:
                logger.warning(f"Not enough data for RSI calculation. Need {period}, got {len(prices)}")
                return pd.Series([50.0] * len(prices), index=prices.index)
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses using exponential moving average
            avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
            avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
            
            # Calculate RS (Relative Strength)
            rs = avg_gains / avg_losses
            
            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))
            
            # Handle edge cases
            rsi = rsi.fillna(50)  # Fill NaN with neutral RSI
            rsi = rsi.replace([np.inf, -np.inf], 50)  # Replace infinite values
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error computing RSI: {e}")
            # Return neutral RSI values on error
            index = data.index if hasattr(data, 'index') else range(len(data))
            return pd.Series([50.0] * len(data), index=index)
    
    @staticmethod
    def compute_sma(data: Union[pd.Series, pd.DataFrame], period: int, column: str = None) -> pd.Series:
        """
        Compute Simple Moving Average (SMA)
        
        Args:
            data: Price data
            period: Moving average period
            column: Column name if DataFrame
            
        Returns:
            pd.Series: SMA values
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            # Calculate SMA
            sma = prices.rolling(window=period, min_periods=1).mean()
            
            return sma
            
        except Exception as e:
            logger.error(f"Error computing SMA: {e}")
            index = data.index if hasattr(data, 'index') else range(len(data))
            return pd.Series([np.nan] * len(data), index=index)
    
    @staticmethod
    def compute_ema(data: Union[pd.Series, pd.DataFrame], period: int, column: str = None) -> pd.Series:
        """
        Compute Exponential Moving Average (EMA)
        
        Args:
            data: Price data
            period: EMA period
            column: Column name if DataFrame
            
        Returns:
            pd.Series: EMA values
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            # Calculate EMA
            ema = prices.ewm(span=period, adjust=False).mean()
            
            return ema
            
        except Exception as e:
            logger.error(f"Error computing EMA: {e}")
            index = data.index if hasattr(data, 'index') else range(len(data))
            return pd.Series([np.nan] * len(data), index=index)
    
    @staticmethod
    def compute_macd(data: Union[pd.Series, pd.DataFrame], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = None) -> dict[str, pd.Series]:
        """
        Compute MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            column: Column name if DataFrame
            
        Returns:
            dict: MACD line, signal line, and histogram
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            # Calculate EMAs
            ema_fast = TechnicalIndicators.compute_ema(prices, fast_period)
            ema_slow = TechnicalIndicators.compute_ema(prices, slow_period)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = TechnicalIndicators.compute_ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"Error computing MACD: {e}")
            index = data.index if hasattr(data, 'index') else range(len(data))
            empty_series = pd.Series([np.nan] * len(data), index=index)
            return {
                'macd': empty_series,
                'signal': empty_series,
                'histogram': empty_series
            }
    
    @staticmethod
    def compute_bollinger_bands(data: Union[pd.Series, pd.DataFrame], period: int = 20, std_dev: float = 2, column: str = None) -> dict[str, pd.Series]:
        """
        Compute Bollinger Bands
        
        Args:
            data: Price data
            period: Moving average period
            std_dev: Standard deviation multiplier
            column: Column name if DataFrame
            
        Returns:
            dict: Upper band, middle band (SMA), lower band
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            # Calculate middle band (SMA)
            middle_band = TechnicalIndicators.compute_sma(prices, period)
            
            # Calculate standard deviation
            rolling_std = prices.rolling(window=period, min_periods=1).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * std_dev)
            lower_band = middle_band - (rolling_std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }
            
        except Exception as e:
            logger.error(f"Error computing Bollinger Bands: {e}")
            index = data.index if hasattr(data, 'index') else range(len(data))
            empty_series = pd.Series([np.nan] * len(data), index=index)
            return {
                'upper': empty_series,
                'middle': empty_series,
                'lower': empty_series
            }
    
    @staticmethod
    def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> dict[str, pd.Series]:
        """
        Compute Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            dict: %K and %D values
        """
        try:
            # Validate input
            if len(high) != len(low) or len(low) != len(close):
                raise ValueError("High, low, and close series must have the same length")
            
            # Calculate %K
            lowest_low = low.rolling(window=k_period, min_periods=1).min()
            highest_high = high.rolling(window=k_period, min_periods=1).max()
            
            # Avoid division by zero
            range_hl = highest_high - lowest_low
            range_hl = range_hl.replace(0, np.nan)
            
            k_percent = 100 * ((close - lowest_low) / range_hl)
            k_percent = k_percent.fillna(50)  # Fill NaN with neutral value
            
            # Calculate %D (SMA of %K)
            d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
            
        except Exception as e:
            logger.error(f"Error computing Stochastic: {e}")
            index = high.index if hasattr(high, 'index') else range(len(high))
            empty_series = pd.Series([np.nan] * len(high), index=index)
            return {
                'k_percent': empty_series,
                'd_percent': empty_series
            }
    
    @staticmethod
    def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Williams %R
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for calculation
            
        Returns:
            pd.Series: Williams %R values
        """
        try:
            # Validate input
            if len(high) != len(low) or len(low) != len(close):
                raise ValueError("High, low, and close series must have the same length")
            
            highest_high = high.rolling(window=period, min_periods=1).max()
            lowest_low = low.rolling(window=period, min_periods=1).min()
            
            # Avoid division by zero
            range_hl = highest_high - lowest_low
            range_hl = range_hl.replace(0, np.nan)
            
            williams_r = -100 * ((highest_high - close) / range_hl)
            williams_r = williams_r.fillna(-50)  # Fill NaN with neutral value
            
            return williams_r
            
        except Exception as e:
            logger.error(f"Error computing Williams %R: {e}")
            index = high.index if hasattr(high, 'index') else range(len(high))
            return pd.Series([-50.0] * len(high), index=index)
    
    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Average True Range (ATR)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            pd.Series: ATR values
        """
        try:
            # Validate input
            if len(high) != len(low) or len(low) != len(close):
                raise ValueError("High, low, and close series must have the same length")
            
            # Calculate True Range components
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            # True Range is the maximum of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR is the exponential moving average of True Range
            atr = true_range.ewm(span=period, adjust=False).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error computing ATR: {e}")
            index = high.index if hasattr(high, 'index') else range(len(high))
            return pd.Series([np.nan] * len(high), index=index)
    
    @staticmethod
    def compute_roc(data: Union[pd.Series, pd.DataFrame], period: int = 12, column: str = None) -> pd.Series:
        """
        Compute Rate of Change (ROC)
        
        Args:
            data: Price data
            period: ROC period
            column: Column name if DataFrame
            
        Returns:
            pd.Series: ROC values
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            # Calculate ROC
            roc = ((prices / prices.shift(period)) - 1) * 100
            
            return roc
            
        except Exception as e:
            logger.error(f"Error computing ROC: {e}")
            index = data.index if hasattr(data, 'index') else range(len(data))
            return pd.Series([np.nan] * len(data), index=index)
    
    @staticmethod
    def compute_momentum(data: Union[pd.Series, pd.DataFrame], period: int = 10, column: str = None) -> pd.Series:
        """
        Compute Momentum
        
        Args:
            data: Price data
            period: Momentum period
            column: Column name if DataFrame
            
        Returns:
            pd.Series: Momentum values
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            # Calculate momentum
            momentum = prices - prices.shift(period)
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error computing Momentum: {e}")
            index = data.index if hasattr(data, 'index') else range(len(data))
            return pd.Series([np.nan] * len(data), index=index)
    
    @staticmethod
    def compute_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Compute Volume Simple Moving Average
        
        Args:
            volume: Volume data
            period: SMA period
            
        Returns:
            pd.Series: Volume SMA
        """
        try:
            if not isinstance(volume, pd.Series):
                raise ValueError("Volume must be a pandas Series")
            
            volume_sma = volume.rolling(window=period, min_periods=1).mean()
            
            return volume_sma
            
        except Exception as e:
            logger.error(f"Error computing Volume SMA: {e}")
            return pd.Series([np.nan] * len(volume), index=volume.index)
    
    @staticmethod
    def detect_support_resistance(data: Union[pd.Series, pd.DataFrame], window: int = 20, column: str = None) -> dict[str, List[Tuple]]:
        """
        Detect support and resistance levels
        
        Args:
            data: Price data
            window: Window for local min/max detection
            column: Column name if DataFrame
            
        Returns:
            dict: Support and resistance levels
        """
        try:
            # Extract the price series
            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column must be specified for DataFrame input")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                prices = data[column].copy()
            else:
                prices = data.copy()
            
            if len(prices) < window * 2 + 1:
                logger.warning(f"Not enough data for support/resistance detection. Need {window*2+1}, got {len(prices)}")
                return {'support': [], 'resistance': []}
            
            support_levels = []
            resistance_levels = []
            
            # Find local minima (support) and maxima (resistance)
            for i in range(window, len(prices) - window):
                window_prices = prices.iloc[i-window:i+window+1]
                current_price = prices.iloc[i]
                
                # Check for local minimum (support)
                if current_price == window_prices.min():
                    support_levels.append((prices.index[i], current_price))
                
                # Check for local maximum (resistance)
                if current_price == window_prices.max():
                    resistance_levels.append((prices.index[i], current_price))
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return {'support': [], 'resistance': []}
    
    @staticmethod
    def validate_ohlc_data(open_prices: pd.Series, high_prices: pd.Series, 
                          low_prices: pd.Series, close_prices: pd.Series) -> bool:
        """
        Validate OHLC data for logical consistency
        
        Args:
            open_prices: Open prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Close prices
            
        Returns:
            bool: True if data is valid
        """
        try:
            # Check if all series have the same length
            lengths = [len(open_prices), len(high_prices), len(low_prices), len(close_prices)]
            if len(set(lengths)) != 1:
                logger.error("OHLC series have different lengths")
                return False
            
            # Check logical consistency
            # High should be >= Open, Close, Low
            if not (high_prices >= open_prices).all():
                logger.error("High prices are not >= Open prices")
                return False
            
            if not (high_prices >= close_prices).all():
                logger.error("High prices are not >= Close prices")
                return False
            
            if not (high_prices >= low_prices).all():
                logger.error("High prices are not >= Low prices")
                return False
            
            # Low should be <= Open, Close, High
            if not (low_prices <= open_prices).all():
                logger.error("Low prices are not <= Open prices")
                return False
            
            if not (low_prices <= close_prices).all():
                logger.error("Low prices are not <= Close prices")
                return False
            
            # Check for negative prices
            for name, series in [('Open', open_prices), ('High', high_prices), ('Low', low_prices), ('Close', close_prices)]:
                if (series < 0).any():
                    logger.error(f"Negative {name} prices found")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating OHLC data: {e}")
            return False

# # utils/__init__.py
# from .indicators import TechnicalIndicators
# from .charts import ChartGenerator

# __all__ = ['TechnicalIndicators', 'ChartGenerator']