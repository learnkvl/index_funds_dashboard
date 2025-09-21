# services/etf_service.py - Complete Enhanced ETF Service with Supabase PostgreSQL Support

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import os
import pickle
from database.db_manager import DatabaseManager
from database.models import ETFDataModel, ETFInfo, ETFAnalysis
from utils.indicators import TechnicalIndicators
from utils.charts import ChartGenerator
from config import Config

logger = logging.getLogger(__name__)

class ETFService:
    """Complete Enhanced ETF Business Logic Service with Supabase PostgreSQL Backend"""
    
    def __init__(self, db_manager: DatabaseManager):
        # === SUPABASE COMPATIBILITY CHECK ===
        try:
            from services.data_service import DataService
        except ImportError as e:
            logger.error(f"Cannot import DataService: {e}")
            logger.error("Please ensure data_service.py exists in the services directory")
            raise ImportError("DataService is required for ETF operations")

        self.db_manager = db_manager
        self.data_service = DataService()
        self.indicators = TechnicalIndicators()
        self.chart_generator = ChartGenerator()
        
        # === SUPABASE CONNECTION VERIFICATION ===
        if not self.db_manager.check_connection():
            raise ConnectionError("Cannot connect to Supabase PostgreSQL database")
        
        logger.info("ETFService initialized with Supabase PostgreSQL backend")
        
        # Initialize ETF list
        self.etf_list = self._load_etf_list()
        self._ensure_etf_info_populated()

    # ============= CORE DATA OPERATIONS =============

    def get_available_etfs(self) -> List[str]:
        """Get list of available ETFs"""
        return self.etf_list
    
    def get_etf_data_smart(self, ticker: str, period: str = '6mo', 
                          force_refresh: bool = False) -> pd.DataFrame:
        """Smart data retrieval - enhanced for Supabase PostgreSQL performance"""
        try:
            # Calculate date range for the period
            end_date = datetime.now()
            start_date = self.data_service.get_period_start_date(period)
            
            # === SUPABASE OPTIMIZED DATA RETRIEVAL ===
            # Get existing data from Supabase with proper date filtering
            existing_data = self.db_manager.get_etf_data(ticker, start_date, end_date)
            
            # Check if we need to fetch more data
            need_fetch = force_refresh or existing_data.empty
            
            if not need_fetch:
                # Check for missing dates
                missing_dates = self.db_manager.get_missing_dates(ticker, start_date, end_date)
                need_fetch = len(missing_dates) > 5  # Fetch if more than 5 days missing
            
            if need_fetch:
                logger.info(f"Fetching fresh data for {ticker} - Supabase backend")
                
                # Fetch fresh data from external API
                fresh_data = self.data_service.fetch_etf_data(ticker, start_date=start_date, end_date=end_date)
                
                if not fresh_data.empty:
                    # Process and add technical indicators
                    processed_data = self._process_etf_data(fresh_data, ticker)
                    
                    # === SUPABASE BATCH INSERT OPTIMIZATION ===
                    # Store in Supabase using optimized batch insert
                    records_inserted = self.db_manager.insert_etf_data(ticker, processed_data)
                    logger.info(f"Inserted {records_inserted} records into Supabase for {ticker}")
                    
                    # Log the refresh in Supabase
                    self.db_manager.log_data_refresh(
                        ticker, 'full_refresh', start_date, end_date, records_inserted
                    )
                    
                    return processed_data
                else:
                    logger.warning(f"No fresh data available for {ticker}")
            
            # Return existing data (add indicators if missing)
            if not existing_data.empty:
                if 'rsi' not in existing_data.columns:
                    logger.info(f"Adding missing technical indicators for {ticker}")
                    existing_data = self._add_technical_indicators(existing_data)
                
                return existing_data
            else:
                logger.warning(f"No data available for {ticker} in Supabase")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting data for {ticker} from Supabase: {e}")
            # === SUPABASE CONNECTION ERROR HANDLING ===
            if "connection" in str(e).lower() or "database" in str(e).lower():
                logger.error("Supabase connection issue detected")
                # Could implement retry logic here
            return pd.DataFrame()
    
    def _process_etf_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Process raw ETF data with technical indicators for Supabase storage"""
        if df.empty:
            return df
        
        # Standardize format
        processed_df = ETFDataModel.from_yfinance_data(ticker, df)
        
        # Add technical indicators
        processed_df = self._add_technical_indicators(processed_df)
        
        # Add trading signals
        processed_df = ETFDataModel.calculate_signals(processed_df)
        
        logger.info(f"Processed {len(processed_df)} records for {ticker} with {len(processed_df.columns)} columns")
        return processed_df

    def force_refresh_etf(self, ticker: str) -> bool:
        """Force refresh ETF data from external source to Supabase"""
        try:
            logger.info(f"Force refreshing {ticker} to Supabase")
            
            # Fetch fresh data for the last year
            fresh_data = self.data_service.fetch_etf_data(ticker, period='1y')
            
            if fresh_data.empty:
                logger.warning(f"No fresh data received for {ticker}")
                return False
            
            # Process the data
            processed_data = self._process_etf_data(fresh_data, ticker)
            
            # Store in Supabase
            records_inserted = self.db_manager.insert_etf_data(ticker, processed_data)
            
            # Log the refresh
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            self.db_manager.log_data_refresh(
                ticker, 'force_refresh', start_date, end_date, records_inserted
            )
            
            logger.info(f"Successfully refreshed {ticker} with {records_inserted} records in Supabase")
            return True
            
        except Exception as e:
            logger.error(f"Error force refreshing {ticker} to Supabase: {e}")
            return False

    # ============= DASHBOARD & OVERVIEW =============

    def get_overview_data(self, period: str = '6mo', threshold: float = 30) -> Optional[dict[str, Any]]:
        """Get overview data for dashboard from Supabase"""
        try:
            # === SUPABASE PERFORMANCE OPTIMIZATION ===
            # Limit ETFs for better Supabase performance
            etfs_to_process = self.etf_list[:Config.MAX_ETFS_OVERVIEW] if hasattr(Config, 'MAX_ETFS_OVERVIEW') else self.etf_list[:50]
            
            all_data = []
            oversold_etfs = []
            
            logger.info(f"Processing overview for {len(etfs_to_process)} ETFs from Supabase")
            
            for ticker in etfs_to_process:
                try:
                    data = self.get_etf_data_smart(ticker, period)
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        current_rsi = latest.get('rsi', 50)
                        
                        etf_summary = {
                            'Ticker': ticker,
                            'RSI': current_rsi,
                            'Close': latest.get('close', 0),
                            'Volume': latest.get('volume', 0),
                            'Date': latest.get('date', datetime.now())
                        }
                        
                        all_data.append(etf_summary)
                        
                        if current_rsi < threshold:
                            oversold_etfs.append(etf_summary)
                
                except Exception as e:
                    logger.warning(f"Error processing {ticker} for overview: {e}")
                    continue
            
            if not all_data:
                logger.warning("No overview data available from Supabase")
                return None
            
            # Create overview chart
            overview_df = pd.DataFrame(all_data)
            chart_json = self.chart_generator.create_overview_chart(overview_df)
            
            logger.info(f"Overview generated: {len(all_data)} ETFs, {len(oversold_etfs)} oversold")
            
            return {
                'chart': chart_json,
                'below_threshold': oversold_etfs,
                'total_etfs': len(all_data),
                'oversold_count': len(oversold_etfs)
            }
            
        except Exception as e:
            logger.error(f"Error generating overview data from Supabase: {e}")
            return None

    def get_etf_detail(self, ticker: str, period: str = '6mo') -> Optional[dict[str, Any]]:
        """Get detailed data for individual ETF with comprehensive technical analysis from Supabase"""
        try:
            # Get ETF data from Supabase
            data = self.get_etf_data_smart(ticker, period)
            
            if data.empty:
                logger.warning(f"No data available for {ticker} in Supabase")
                return None
            
            # Get ETF info from Supabase
            etf_info = self.db_manager.get_etf_info(ticker)
            if etf_info.empty:
                etf_info_dict = {'name': ticker, 'aum': 'N/A', 'ticker': ticker}
            else:
                info_row = etf_info.iloc[0]
                aum_formatted = self._format_aum(info_row.get('aum', 0))
                etf_info_dict = {
                    'name': info_row.get('name', ticker),
                    'aum': aum_formatted,
                    'ticker': ticker
                }
            
            # Generate charts
            price_chart = self.chart_generator.create_price_chart(data, ticker)
            rsi_chart = self.chart_generator.create_rsi_chart(data, ticker)
            
            # Get latest data and analysis
            latest = data.iloc[-1]
            analysis = ETFDataModel.get_latest_analysis(data, ticker)
            
            # Extract all technical indicators
            latest_indicators = {
                'price': round(latest.get('close', 0), 2),
                'rsi': round(latest.get('rsi', 50), 2),
                'volume': int(latest.get('volume', 0)),
                'ema_50': round(latest.get('ema_50', 0), 2) if latest.get('ema_50') else None,
                'ema_200': round(latest.get('ema_200', 0), 2) if latest.get('ema_200') else None,
                'macd': round(latest.get('macd', 0), 4) if latest.get('macd') else None,
                'macd_signal': round(latest.get('macd_signal', 0), 4) if latest.get('macd_signal') else None,
                'macd_histogram': round(latest.get('macd_histogram', 0), 4) if latest.get('macd_histogram') else None,
                'supertrend': round(latest.get('supertrend', 0), 2) if latest.get('supertrend') else None,
                'adx': round(latest.get('adx', 0), 2) if latest.get('adx') else None,
                'volume_sma': round(latest.get('volume_sma', 0), 0) if latest.get('volume_sma') else None,
                'date': latest.get('date', datetime.now()).strftime('%Y-%m-%d') if pd.notna(latest.get('date')) else datetime.now().strftime('%Y-%m-%d')
            }
            
            # Generate comprehensive technical analysis
            technical_analysis = self._generate_comprehensive_analysis(latest_indicators)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(latest_indicators)
            
            # Get enhanced signal from our multi-factor analysis
            enhanced_signal = self._determine_enhanced_signal(latest, data)
            signal_strength = self._calculate_signal_strength(latest, data)
            signal_components = self._get_signal_components(latest)
            
            logger.info(f"ETF detail generated for {ticker} with enhanced analysis")
            
            return {
                'price_chart': price_chart,
                'rsi_chart': rsi_chart,
                'latest_data': latest_indicators,
                'etf_info': etf_info_dict,
                'analysis': analysis.__dict__ if analysis else None,
                'technical_analysis': technical_analysis,
                'trading_signals': trading_signals,
                'enhanced_signal': {
                    'signal': enhanced_signal,
                    'strength': signal_strength,
                    'components': signal_components
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting ETF detail for {ticker} from Supabase: {e}")
            return None

    def _generate_comprehensive_analysis(self, indicators: dict) -> dict:
        """Generate comprehensive technical analysis for all indicators"""
        analysis = {}
        
        # RSI Analysis
        rsi = indicators.get('rsi')
        if rsi:
            if rsi < 30:
                analysis['rsi'] = {
                    'condition': 'Oversold Condition',
                    'level': f"{rsi:.2f} (Oversold)",
                    'sentiment': 'Bearish, but potential reversal',
                    'risk_level': 'Moderate to High',
                    'description': f"The RSI is currently at {rsi:.2f}, indicating the ETF may be oversold. This could present a potential buying opportunity as the price may be due for a rebound."
                }
            elif rsi > 70:
                analysis['rsi'] = {
                    'condition': 'Overbought Condition',
                    'level': f"{rsi:.2f} (Overbought)",
                    'sentiment': 'Bullish, but potential correction',
                    'risk_level': 'High',
                    'description': f"The RSI is currently at {rsi:.2f}, indicating the ETF may be overbought. This suggests the price may be due for a pullback or consolidation."
                }
            else:
                analysis['rsi'] = {
                    'condition': 'Normal Trading Range',
                    'level': f"{rsi:.2f} (Normal)",
                    'sentiment': 'Neutral',
                    'risk_level': 'Moderate',
                    'description': f"The RSI is currently at {rsi:.2f}, indicating normal trading conditions. No immediate overbought or oversold signals are present."
                }
        
        # MACD Analysis
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        macd_histogram = indicators.get('macd_histogram')
        
        if macd and macd_signal:
            if macd > macd_signal:
                trend_direction = "bullish momentum"
                signal_status = "Bullish Crossover"
                sentiment = "Bullish momentum building"
                risk_level = "Low to Moderate"
            else:
                trend_direction = "bearish momentum"
                signal_status = "Bearish Crossover"
                sentiment = "Bearish momentum building"
                risk_level = "Moderate to High"
            
            histogram_trend = "strengthening" if macd_histogram and macd_histogram > 0 else "weakening"
            
            analysis['macd'] = {
                'condition': signal_status,
                'level': f"MACD: {macd:.4f}, Signal: {macd_signal:.4f}",
                'sentiment': sentiment,
                'risk_level': risk_level,
                'description': f"The MACD line ({macd:.4f}) is {'above' if macd > macd_signal else 'below'} the signal line ({macd_signal:.4f}), indicating {trend_direction}. The histogram shows momentum is {histogram_trend}."
            }
        
        # EMA 50/200 Analysis (Golden Cross / Death Cross)
        ema_50 = indicators.get('ema_50')
        ema_200 = indicators.get('ema_200')
        
        if ema_50 and ema_200:
            gap_percent = ((ema_50 - ema_200) / ema_200) * 100
            
            if ema_50 > ema_200:
                analysis['ema_cross'] = {
                    'condition': 'Golden Cross Active',
                    'level': f"EMA 50: ${ema_50:.2f}, EMA 200: ${ema_200:.2f}",
                    'sentiment': 'Bullish long-term trend',
                    'risk_level': 'Low to Moderate',
                    'description': f"The 50-day EMA (${ema_50:.2f}) is above the 200-day EMA (${ema_200:.2f}) by {gap_percent:.2f}%, confirming a bullish long-term trend. This Golden Cross suggests continued upward momentum."
                }
            else:
                analysis['ema_cross'] = {
                    'condition': 'Death Cross Active',
                    'level': f"EMA 50: ${ema_50:.2f}, EMA 200: ${ema_200:.2f}",
                    'sentiment': 'Bearish long-term trend',
                    'risk_level': 'High',
                    'description': f"The 50-day EMA (${ema_50:.2f}) is below the 200-day EMA (${ema_200:.2f}) by {abs(gap_percent):.2f}%, indicating a bearish long-term trend. This Death Cross suggests continued downward pressure."
                }
        
        # Supertrend Analysis
        supertrend = indicators.get('supertrend')
        current_price = indicators.get('price')
        
        if supertrend and current_price:
            distance_percent = ((current_price - supertrend) / supertrend) * 100
            
            if current_price > supertrend:
                analysis['supertrend'] = {
                    'condition': 'Bullish Supertrend',
                    'level': f"Price: ${current_price:.2f}, Supertrend: ${supertrend:.2f}",
                    'sentiment': 'Bullish trend confirmed',
                    'risk_level': 'Low to Moderate',
                    'description': f"The current price (${current_price:.2f}) is {distance_percent:.2f}% above the Supertrend line (${supertrend:.2f}), confirming the bullish trend. The trend is likely to continue upward."
                }
            else:
                analysis['supertrend'] = {
                    'condition': 'Bearish Supertrend',
                    'level': f"Price: ${current_price:.2f}, Supertrend: ${supertrend:.2f}",
                    'sentiment': 'Bearish trend confirmed',
                    'risk_level': 'Moderate to High',
                    'description': f"The current price (${current_price:.2f}) is {abs(distance_percent):.2f}% below the Supertrend line (${supertrend:.2f}), confirming the bearish trend. The trend is likely to continue downward."
                }
        
        # ADX Analysis
        adx = indicators.get('adx')
        
        if adx:
            if adx > 25:
                strength = "Strong"
                description = f"The ADX is at {adx:.2f}, indicating a strong trending market. Trend-following strategies are more reliable in this environment."
                risk_level = "Low to Moderate"
            elif adx > 20:
                strength = "Moderate"
                description = f"The ADX is at {adx:.2f}, indicating a moderate trend. Some trend-following signals may be reliable, but exercise caution."
                risk_level = "Moderate"
            else:
                strength = "Weak"
                description = f"The ADX is at {adx:.2f}, indicating a weak or ranging market. Trend-following strategies may be less reliable."
                risk_level = "High for trend trades"
            
            analysis['adx'] = {
                'condition': f'{strength} Trend Strength',
                'level': f"ADX: {adx:.2f}",
                'sentiment': f'{strength} directional movement',
                'risk_level': risk_level,
                'description': description
            }
        
        # Volume Analysis
        volume = indicators.get('volume')
        volume_sma = indicators.get('volume_sma')
        
        if volume and volume_sma:
            volume_ratio = volume / volume_sma
            
            if volume_ratio > 1.5:
                analysis['volume'] = {
                    'condition': 'High Volume Activity',
                    'level': f"Volume: {volume:,}, Avg: {volume_sma:,.0f}",
                    'sentiment': 'Strong market interest',
                    'risk_level': 'Low (high conviction)',
                    'description': f"Current volume ({volume:,}) is {volume_ratio:.1f}x the average ({volume_sma:,.0f}), indicating strong market interest and conviction behind price movements."
                }
            elif volume_ratio < 0.7:
                analysis['volume'] = {
                    'condition': 'Low Volume Activity',
                    'level': f"Volume: {volume:,}, Avg: {volume_sma:,.0f}",
                    'sentiment': 'Weak market interest',
                    'risk_level': 'High (low conviction)',
                    'description': f"Current volume ({volume:,}) is only {volume_ratio:.1f}x the average ({volume_sma:,.0f}), indicating weak market interest. Price movements may be less reliable."
                }
            else:
                analysis['volume'] = {
                    'condition': 'Normal Volume Activity',
                    'level': f"Volume: {volume:,}, Avg: {volume_sma:,.0f}",
                    'sentiment': 'Average market interest',
                    'risk_level': 'Moderate',
                    'description': f"Current volume ({volume:,}) is near average ({volume_sma:,.0f}), indicating normal market participation."
                }
        
        return analysis

    def _generate_trading_signals(self, indicators: dict) -> dict:
        """Generate comprehensive trading signals based on all indicators"""
        signals = {}
        
        # RSI Signals
        rsi = indicators.get('rsi')
        if rsi:
            if rsi < 30:
                signals['rsi'] = {
                    'signal': 'Strong Buy Signal',
                    'status': 'buy',
                    'actions': [
                        'RSI indicates oversold condition',
                        'Wait for confirmation of trend reversal',
                        'Consider dollar-cost averaging',
                        'Set stop-loss at recent support levels'
                    ]
                }
            elif rsi > 70:
                signals['rsi'] = {
                    'signal': 'Strong Sell Signal',
                    'status': 'sell',
                    'actions': [
                        'RSI indicates overbought condition',
                        'Consider taking profits',
                        'Reduce position size',
                        'Watch for bearish divergence'
                    ]
                }
            else:
                signals['rsi'] = {
                    'signal': 'Hold Signal',
                    'status': 'hold',
                    'actions': [
                        'RSI in normal range',
                        'Monitor for trend changes',
                        'Wait for clearer signals',
                        'Consider trend-following strategy'
                    ]
                }
        
        # MACD Signals
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        
        if macd and macd_signal:
            if macd > macd_signal:
                signals['macd'] = {
                    'signal': 'Bullish MACD Signal',
                    'status': 'buy',
                    'actions': [
                        'MACD crossed above signal line',
                        'Momentum is building upward',
                        'Consider entering long position',
                        'Monitor for continued strength'
                    ]
                }
            else:
                signals['macd'] = {
                    'signal': 'Bearish MACD Signal',
                    'status': 'sell',
                    'actions': [
                        'MACD crossed below signal line',
                        'Momentum is building downward',
                        'Consider exiting long positions',
                        'Watch for further weakness'
                    ]
                }
        
        # EMA Cross Signals
        ema_50 = indicators.get('ema_50')
        ema_200 = indicators.get('ema_200')
        
        if ema_50 and ema_200:
            if ema_50 > ema_200:
                signals['ema_cross'] = {
                    'signal': 'Golden Cross - Buy Signal',
                    'status': 'buy',
                    'actions': [
                        'Long-term bullish trend confirmed',
                        'Suitable for trend-following strategies',
                        'Consider increasing position size',
                        'Use 200-day EMA as support level'
                    ]
                }
            else:
                signals['ema_cross'] = {
                    'signal': 'Death Cross - Sell Signal',
                    'status': 'sell',
                    'actions': [
                        'Long-term bearish trend confirmed',
                        'Avoid new long positions',
                        'Consider defensive positioning',
                        'Use 200-day EMA as resistance level'
                    ]
                }
        
        # Supertrend Signals
        supertrend = indicators.get('supertrend')
        current_price = indicators.get('price')
        
        if supertrend and current_price:
            if current_price > supertrend:
                signals['supertrend'] = {
                    'signal': 'Supertrend Buy Signal',
                    'status': 'buy',
                    'actions': [
                        'Price above Supertrend line',
                        'Trend is bullish',
                        'Use Supertrend as trailing stop',
                        'Suitable for trend-following'
                    ]
                }
            else:
                signals['supertrend'] = {
                    'signal': 'Supertrend Sell Signal',
                    'status': 'sell',
                    'actions': [
                        'Price below Supertrend line',
                        'Trend is bearish',
                        'Avoid long positions',
                        'Consider short opportunities'
                    ]
                }
        
        # ADX Signals
        adx = indicators.get('adx')
        if adx:
            if adx > 25:
                signals['adx'] = {
                    'signal': 'Strong Trend Confirmed',
                    'status': 'trend_strong',
                    'actions': [
                        'Strong trending market identified',
                        'Trend-following strategies recommended',
                        'High probability of trend continuation',
                        'Avoid counter-trend trades'
                    ]
                }
            else:
                signals['adx'] = {
                    'signal': 'Weak Trend - Range Bound',
                    'status': 'trend_weak',
                    'actions': [
                        'Weak trending market',
                        'Consider range-trading strategies',
                        'Trend-following less reliable',
                        'Wait for stronger directional movement'
                    ]
                }
        
        # Volume Signals
        volume = indicators.get('volume')
        volume_sma = indicators.get('volume_sma')
        
        if volume and volume_sma:
            volume_ratio = volume / volume_sma
            if volume_ratio > 1.5:
                signals['volume'] = {
                    'signal': 'High Volume Confirmation',
                    'status': 'confirmation',
                    'actions': [
                        'High volume supports price movement',
                        'Strong market conviction',
                        'Signals are more reliable',
                        'Consider increasing position size'
                    ]
                }
            elif volume_ratio < 0.7:
                signals['volume'] = {
                    'signal': 'Low Volume Warning',
                    'status': 'warning',
                    'actions': [
                        'Low volume questions price movement',
                        'Weak market conviction',
                        'Signals are less reliable',
                        'Exercise extra caution'
                    ]
                }
        
        return signals

    # ============= FILTERING & ANALYSIS =============

    def get_etfs_by_criteria(self, 
                           rsi_min: Optional[float] = None,
                           rsi_max: Optional[float] = None,
                           volume_min: Optional[int] = None,
                           price_min: Optional[float] = None,
                           price_max: Optional[float] = None,
                           period: str = '6mo') -> List[dict[str, Any]]:
        """Get ETFs matching specific criteria from Supabase"""
        try:
            matching_etfs = []
            
            logger.info(f"Filtering ETFs with criteria: RSI({rsi_min}-{rsi_max}), Price({price_min}-{price_max}), Volume({volume_min})")
            
            for ticker in self.etf_list:
                try:
                    data = self.get_etf_data_smart(ticker, period)
                    
                    if data.empty:
                        continue
                    
                    latest = data.iloc[-1]
                    current_rsi = latest.get('rsi', 50)
                    current_price = latest.get('close', 0)
                    current_volume = latest.get('volume', 0)
                    
                    # Apply filters
                    if rsi_min is not None and current_rsi < rsi_min:
                        continue
                    if rsi_max is not None and current_rsi > rsi_max:
                        continue
                    if volume_min is not None and current_volume < volume_min:
                        continue
                    if price_min is not None and current_price < price_min:
                        continue
                    if price_max is not None and current_price > price_max:
                        continue
                    
                    # Get ETF info from Supabase
                    etf_info = self.db_manager.get_etf_info(ticker)
                    name = etf_info.iloc[0]['name'] if not etf_info.empty else ticker
                    
                    matching_etfs.append({
                        'ticker': ticker,
                        'name': name,
                        'rsi': round(current_rsi, 2),
                        'price': round(current_price, 2),
                        'volume': current_volume,
                        'date': latest.get('date', datetime.now()).strftime('%Y-%m-%d')
                    })
                    
                except Exception as e:
                    logger.warning(f"Error checking criteria for {ticker}: {e}")
                    continue
            
            logger.info(f"Found {len(matching_etfs)} ETFs matching criteria")
            return matching_etfs
            
        except Exception as e:
            logger.error(f"Error filtering ETFs by criteria: {e}")
            return []

    def get_data_quality_report(self, ticker: str, period: str = '1y') -> dict[str, Any]:
        """Get data quality report for an ETF from Supabase"""
        try:
            data = self.get_etf_data_smart(ticker, period)
            quality_metrics = ETFDataModel.validate_data_quality(data)
            
            # Add additional metrics
            latest_date = self.db_manager.get_latest_date(ticker)
            if latest_date:
                days_since_update = (datetime.now() - latest_date).days
                quality_metrics['days_since_update'] = days_since_update
                quality_metrics['is_current'] = days_since_update <= 3
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error generating quality report for {ticker}: {e}")
            return {'is_valid': False, 'error': str(e)}

    # ============= ENHANCED SIGNAL METHODS =============

    def get_all_etf_signals(self, period: str = '6mo') -> list:
        """Return all ETFs with enhanced technical indicators and signals - optimized for Supabase"""
        results = []
        
        # === SUPABASE PERFORMANCE OPTIMIZATION ===
        # Limit ETFs for better Supabase performance and process in batches
        etfs_to_process = self.etf_list[:Config.MAX_ETFS_OVERVIEW] if hasattr(Config, 'MAX_ETFS_OVERVIEW') else self.etf_list[:50]
        
        logger.info(f"Processing {len(etfs_to_process)} ETFs for enhanced signals from Supabase")
        
        # Process in smaller batches for better Supabase connection management
        batch_size = 10  # Process 10 ETFs at a time
        for i in range(0, len(etfs_to_process), batch_size):
            batch = etfs_to_process[i:i + batch_size]
            logger.info(f"Processing ETF batch {i//batch_size + 1}/{(len(etfs_to_process) + batch_size - 1)//batch_size}: {batch}")
            
            for ticker in batch:
                try:
                    # Get ETF data with all technical indicators from Supabase
                    data = self.get_etf_data_smart(ticker, period)
                    
                    if data.empty:
                        logger.warning(f"No data available for {ticker}")
                        continue
                    
                    # Get the latest row
                    latest = data.iloc[-1]
                    
                    # Extract all technical indicators with proper error handling
                    result = {
                        'ticker': ticker,
                        # Trend Indicators
                        'ema_50': self._safe_float(latest.get('ema_50')),
                        'ema_200': self._safe_float(latest.get('ema_200')),
                        'sma_50': self._safe_float(latest.get('sma_50')),
                        'supertrend': self._safe_float(latest.get('supertrend')),
                        
                        # Momentum Indicators
                        'rsi': self._safe_float(latest.get('rsi')),
                        'macd': self._safe_float(latest.get('macd')),
                        'macd_signal': self._safe_float(latest.get('macd_signal')),
                        'macd_histogram': self._safe_float(latest.get('macd_histogram')),
                        
                        # Strength Indicators
                        'adx': self._safe_float(latest.get('adx')),
                        'volume': self._safe_int(latest.get('volume')),
                        'volume_sma': self._safe_float(latest.get('volume_sma')),
                        
                        # Price Data
                        'close_price': self._safe_float(latest.get('close')),
                        'high': self._safe_float(latest.get('high')),
                        'low': self._safe_float(latest.get('low')),
                        
                        # Enhanced Signal
                        'signal': self._determine_enhanced_signal(latest, data),
                        'signal_strength': self._calculate_signal_strength(latest, data),
                        'signal_components': self._get_signal_components(latest),
                        
                        'date': latest.get('date', datetime.now()).strftime('%Y-%m-%d') if pd.notna(latest.get('date')) else datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    results.append(result)
                    logger.debug(f"Processed {ticker}: Signal={result['signal']}, Strength={result['signal_strength']}")
                    
                except Exception as e:
                    logger.warning(f"Error processing {ticker} for signals: {e}")
                    # Add placeholder data to show the ticker exists but has issues
                    results.append({
                        'ticker': ticker,
                        'ema_50': None, 'ema_200': None, 'sma_50': None, 'supertrend': None,
                        'rsi': None, 'macd': None, 'macd_signal': None, 'macd_histogram': None,
                        'adx': None, 'volume': None, 'volume_sma': None,
                        'close_price': None, 'high': None, 'low': None,
                        'signal': 'ERROR', 'signal_strength': 'WEAK',
                        'signal_components': {}, 'date': datetime.now().strftime('%Y-%m-%d')
                    })
                    continue
        
        logger.info(f"Processed {len(results)} ETFs for enhanced signals using Supabase")
        return results

    # ============= ENHANCED SIGNAL LOGIC =============

    def _determine_enhanced_signal(self, latest_data, historical_data=None):
        """
        Enhanced signal determination using multi-factor analysis:
        1. Trend Analysis (EMA 50/200, Supertrend)
        2. Momentum Indicators (RSI, MACD)
        3. Volume & Strength (Volume SMA, ADX)
        """
        try:
            # Extract indicators
            ema_50 = self._safe_float(latest_data.get('ema_50'))
            ema_200 = self._safe_float(latest_data.get('ema_200'))
            supertrend = self._safe_float(latest_data.get('supertrend'))
            close_price = self._safe_float(latest_data.get('close'))
            
            rsi = self._safe_float(latest_data.get('rsi'))
            macd = self._safe_float(latest_data.get('macd'))
            macd_signal = self._safe_float(latest_data.get('macd_signal'))
            
            adx = self._safe_float(latest_data.get('adx'))
            volume = self._safe_int(latest_data.get('volume'))
            volume_sma = self._safe_float(latest_data.get('volume_sma'))
            
            # Check for missing essential data
            if None in [close_price, ema_50, ema_200, rsi]:
                return 'HOLD'
            
            # --- 1. TREND ANALYSIS ---
            trend_score = 0
            
            # Golden Cross / Death Cross (EMA 50 vs EMA 200)
            if ema_50 > ema_200:
                trend_score += 2  # Strong positive trend
            else:
                trend_score -= 2  # Negative trend (Death Cross)
            
            # Supertrend confirmation
            if supertrend and close_price:
                if close_price > supertrend:
                    trend_score += 1  # Price above supertrend
                else:
                    trend_score -= 1  # Price below supertrend
            
            # --- 2. MOMENTUM ANALYSIS ---
            momentum_score = 0
            
            # RSI - Sweet spot between 40-60, avoid extremes
            if 40 <= rsi <= 60:
                momentum_score += 2  # Ideal momentum zone
            elif 30 <= rsi < 40:
                momentum_score += 1  # Moderate oversold, could be good entry
            elif 60 < rsi <= 70:
                momentum_score -= 1  # Getting overbought, caution
            elif rsi > 70:
                momentum_score -= 2  # Overbought, avoid
            elif rsi < 30:
                momentum_score += 0  # Oversold but wait for confirmation
            
            # MACD Crossover
            if macd and macd_signal:
                if macd > macd_signal:
                    momentum_score += 1  # MACD above signal line
                else:
                    momentum_score -= 1  # MACD below signal line
            
            # --- 3. VOLUME & STRENGTH ANALYSIS ---
            strength_score = 0
            
            # ADX Strength
            if adx:
                if adx > 25:
                    strength_score += 1  # Strong trend confirmed
                else:
                    strength_score -= 1  # Weak trend, avoid
            
            # Volume confirmation
            if volume and volume_sma:
                if volume > volume_sma * 1.2:  # 20% above average
                    strength_score += 1  # High volume confirms move
                elif volume < volume_sma * 0.8:  # 20% below average
                    strength_score -= 1  # Low volume, weak signal
            
            # --- FINAL SIGNAL DETERMINATION ---
            total_score = trend_score + momentum_score + strength_score
            
            # Decision logic with strict criteria
            if total_score >= 4 and trend_score >= 1:  # Strong buy conditions
                return 'STRONG_BUY'
            elif total_score >= 2 and trend_score >= 0:  # Good buy conditions
                return 'BUY'
            elif total_score <= -4 or trend_score <= -2:  # Strong sell conditions
                return 'STRONG_SELL'
            elif total_score <= -2:  # Sell conditions
                return 'SELL'
            else:
                return 'HOLD'  # Neutral conditions
                
        except Exception as e:
            logger.warning(f"Error in enhanced signal determination: {e}")
            return 'HOLD'

    def _calculate_signal_strength(self, latest_data, historical_data=None):
        """Calculate signal strength based on conviction level"""
        try:
            # Count confirming indicators
            confirmations = 0
            total_indicators = 0
            
            # Trend confirmations
            ema_50 = self._safe_float(latest_data.get('ema_50'))
            ema_200 = self._safe_float(latest_data.get('ema_200'))
            if ema_50 and ema_200:
                total_indicators += 1
                if ema_50 > ema_200:
                    confirmations += 1
            
            # Momentum confirmations
            rsi = self._safe_float(latest_data.get('rsi'))
            if rsi:
                total_indicators += 1
                if 40 <= rsi <= 60:
                    confirmations += 1
            
            macd = self._safe_float(latest_data.get('macd'))
            macd_signal = self._safe_float(latest_data.get('macd_signal'))
            if macd and macd_signal:
                total_indicators += 1
                if macd > macd_signal:
                    confirmations += 1
            
            # Strength confirmations
            adx = self._safe_float(latest_data.get('adx'))
            if adx:
                total_indicators += 1
                if adx > 25:
                    confirmations += 1
            
            if total_indicators == 0:
                return 'WEAK'
            
            conviction_ratio = confirmations / total_indicators
            
            if conviction_ratio >= 0.75:
                return 'STRONG'
            elif conviction_ratio >= 0.5:
                return 'MODERATE'
            else:
                return 'WEAK'
                
        except Exception as e:
            logger.warning(f"Error calculating signal strength: {e}")
            return 'WEAK'

    def _get_signal_components(self, latest_data):
        """Get detailed breakdown of signal components for analysis"""
        try:
            components = {}
            
            # Trend components
            ema_50 = self._safe_float(latest_data.get('ema_50'))
            ema_200 = self._safe_float(latest_data.get('ema_200'))
            
            if ema_50 and ema_200:
                components['trend'] = 'BULLISH' if ema_50 > ema_200 else 'BEARISH'
                components['ema_gap'] = round(((ema_50 - ema_200) / ema_200) * 100, 2)
            
            # Momentum components
            rsi = self._safe_float(latest_data.get('rsi'))
            if rsi:
                if 40 <= rsi <= 60:
                    components['rsi_zone'] = 'IDEAL'
                elif rsi < 30:
                    components['rsi_zone'] = 'OVERSOLD'
                elif rsi > 70:
                    components['rsi_zone'] = 'OVERBOUGHT'
                else:
                    components['rsi_zone'] = 'NEUTRAL'
            
            # MACD status
            macd = self._safe_float(latest_data.get('macd'))
            macd_signal = self._safe_float(latest_data.get('macd_signal'))
            if macd and macd_signal:
                components['macd_status'] = 'BULLISH' if macd > macd_signal else 'BEARISH'
            
            # Strength components
            adx = self._safe_float(latest_data.get('adx'))
            if adx:
                components['trend_strength'] = 'STRONG' if adx > 25 else 'WEAK'
            
            return components
            
        except Exception as e:
            logger.warning(f"Error getting signal components: {e}")
            return {}

    # ============= TECHNICAL INDICATORS =============

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators including EMA 50/200 - optimized for Supabase storage"""
        if df.empty:
            return df
        
        try:
            df = df.copy()
            logger.info(f"Adding enhanced technical indicators to {len(df)} rows of data")
            
            # Basic RSI indicators
            if 'close' in df.columns:
                df['rsi'] = self.indicators.compute_rsi(df, column='close')
                logger.info("Added RSI indicator")
            
            if 'volume' in df.columns:
                df['rsi_vol'] = self.indicators.compute_rsi(df, column='volume')
                logger.info("Added Volume RSI indicator")
            
            # Enhanced Moving averages (including 50 and 200 period EMAs)
            if 'close' in df.columns:
                df['sma_20'] = self.indicators.compute_sma(df, period=20, column='close')
                df['sma_50'] = self.indicators.compute_sma(df, period=50, column='close')
                df['ema_12'] = self.indicators.compute_ema(df, period=12, column='close')
                df['ema_26'] = self.indicators.compute_ema(df, period=26, column='close')
                
                # Add the critical EMA 50 and EMA 200 for signal generation
                df['ema_50'] = self.indicators.compute_ema(df, period=50, column='close')
                df['ema_200'] = self.indicators.compute_ema(df, period=200, column='close')
                
                logger.info("Added enhanced moving averages including EMA 50/200")
            
            # MACD system
            if 'close' in df.columns:
                macd_data = self.indicators.compute_macd(df, column='close')
                df['macd'] = macd_data['macd']
                df['macd_signal'] = macd_data['signal']
                df['macd_histogram'] = macd_data['histogram']
                logger.info("Added MACD indicators")
            
            # Bollinger Bands
            if 'close' in df.columns:
                bb_data = self.indicators.compute_bollinger_bands(df, column='close')
                df['bb_upper'] = bb_data['upper']
                df['bb_middle'] = bb_data['middle']
                df['bb_lower'] = bb_data['lower']
                logger.info("Added Bollinger Bands")
            
            # Advanced indicators (require OHLC data)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                # ATR
                df['atr'] = self.indicators.compute_atr(df['high'], df['low'], df['close'])
                
                # Stochastic Oscillator
                stoch_data = self.indicators.compute_stochastic(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch_data['k_percent']
                df['stoch_d'] = stoch_data['d_percent']
                
                # Williams %R
                df['williams_r'] = self.indicators.compute_williams_r(df['high'], df['low'], df['close'])
                
                # ADX
                df['adx'] = self.indicators.compute_adx(df)
                
                # Supertrend
                df['supertrend'] = self.indicators.compute_supertrend(df)
                
                # Parabolic SAR
                df['parabolic_sar'] = self.indicators.compute_parabolic_sar(df)
                
                # Ichimoku Cloud components
                ichimoku_data = self.indicators.compute_ichimoku(df)
                df['tenkan_sen'] = ichimoku_data['tenkan_sen']
                df['kijun_sen'] = ichimoku_data['kijun_sen']
                df['senkou_span_a'] = ichimoku_data['senkou_span_a']
                df['senkou_span_b'] = ichimoku_data['senkou_span_b']
                df['chikou_span'] = ichimoku_data['chikou_span']
                
                logger.info("Added advanced OHLC indicators")
            else:
                logger.warning("Missing OHLC data for advanced indicators")
            
            # Rate of Change and Momentum
            if 'close' in df.columns:
                df['roc'] = self.indicators.compute_roc(df, column='close')
                df['momentum'] = self.indicators.compute_momentum(df, column='close')
                logger.info("Added ROC and Momentum")
            
            # Linear Regression
            if 'close' in df.columns:
                df['linear_regression'] = self.indicators.compute_linear_regression(df, column='close')
                logger.info("Added Linear Regression")
            
            # Enhanced Volume Analysis
            if 'volume' in df.columns:
                df['volume_sma'] = self.indicators.compute_volume_sma(df['volume'])
                logger.info("Added Volume SMA")
            
            # Add enhanced trading signal based on new logic
            if all(col in df.columns for col in ['rsi', 'ema_50', 'ema_200']):
                df['signal'] = df.apply(lambda row: self._determine_enhanced_signal(row), axis=1)
                logger.info("Added enhanced trading signals")
            else:
                # Fallback to basic RSI signal
                if 'rsi' in df.columns:
                    df['signal'] = df['rsi'].apply(lambda x: 
                        'BUY' if x < 30 else 
                        'SELL' if x > 70 else 
                        'HOLD' if pd.notna(x) else 'HOLD'
                    )
                    logger.info("Added basic RSI trading signals")
            
            # Clean up any infinite or extremely large values before Supabase storage
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Successfully added all enhanced technical indicators. DataFrame now has {len(df.columns)} columns ready for Supabase")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            # Return original DataFrame if indicator calculation fails
            return df

    # ============= UTILITY METHODS =============

    def _safe_float(self, value):
        """Safely convert value to float, return None if not possible"""
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_int(self, value):
        """Safely convert value to int, return None if not possible"""
        try:
            if pd.isna(value):
                return None
            return int(value)
        except (ValueError, TypeError):
            return None

    def _format_aum(self, aum: float) -> str:
        """Format AUM for display"""
        if aum == 0 or pd.isna(aum):
            return 'N/A'
        
        if aum >= 1e12:
            return f"${aum/1e12:.1f}T"
        elif aum >= 1e9:
            return f"${aum/1e9:.1f}B"
        elif aum >= 1e6:
            return f"${aum/1e6:.1f}M"
        else:
            return f"${aum:,.0f}"

    # ============= INITIALIZATION METHODS =============
    
    def _load_etf_list(self) -> List[str]:
        """Load ETF list from various sources"""
        etfs = []
        
        # Try to load from CSV file
        if os.path.exists(Config.ETF_LIST_CSV):
            try:
                df = pd.read_csv(Config.ETF_LIST_CSV)
                if 'Symbol' in df.columns:
                    etfs.extend(df['Symbol'].tolist())
                    logger.info(f"Loaded {len(etfs)} ETFs from CSV")
            except Exception as e:
                logger.warning(f"Error loading ETF list from CSV: {e}")
        
        # Try to load from pickle file
        if os.path.exists(Config.ETF_INFO_PICKLE) and not etfs:
            try:
                with open(Config.ETF_INFO_PICKLE, 'rb') as f:
                    pickle_data = pickle.load(f)
                    for item in pickle_data:
                        if isinstance(item, dict) and 'ticker' in item:
                            etfs.append(item['ticker'])
                    logger.info(f"Loaded {len(etfs)} ETFs from pickle")
            except Exception as e:
                logger.warning(f"Error loading ETF list from pickle: {e}")
        
        # Fallback to default list
        if not etfs:
            etfs = Config.DEFAULT_ETF_LIST
            logger.info(f"Using default ETF list with {len(etfs)} ETFs")
        
        return list(set(etfs))  # Remove duplicates
    
    def _ensure_etf_info_populated(self):
        """Ensure ETF info is populated in Supabase database"""
        try:
            # === SUPABASE CONNECTION CHECK ===
            if not self.db_manager.check_connection():
                logger.error("Cannot connect to Supabase for ETF info population")
                return
            
            existing_info = self.db_manager.get_etf_info()
            existing_tickers = set(existing_info['ticker'].tolist()) if not existing_info.empty else set()
            
            # Find ETFs without info
            missing_info = [ticker for ticker in self.etf_list if ticker not in existing_tickers]
            
            if missing_info:
                logger.info(f"Populating info for {len(missing_info)} ETFs in Supabase")
                
                # Try to load from existing files first
                self._load_etf_info_from_files()
                
                # === SUPABASE RATE LIMITING ===
                # Then fetch missing ones from API (limit to avoid overwhelming Supabase)
                still_missing = [ticker for ticker in missing_info[:10]]  # Limit to 10 for initial load
                for ticker in still_missing:
                    try:
                        info = self.data_service.fetch_etf_info(ticker)
                        self.db_manager.upsert_etf_info(
                            ticker=ticker,
                            name=info.get('name'),
                            aum=info.get('aum'),
                            description=info.get('description')
                        )
                        logger.debug(f"Updated Supabase with info for {ticker}")
                    except Exception as e:
                        logger.warning(f"Could not fetch info for {ticker}: {e}")
                        # Insert basic info
                        self.db_manager.upsert_etf_info(ticker=ticker, name=ticker)
                        
        except Exception as e:
            logger.error(f"Error populating ETF info in Supabase: {e}")
    
    def _load_etf_info_from_files(self):
        """Load ETF info from existing files into Supabase"""
        # Load from CSV
        if os.path.exists(Config.ETF_LIST_CSV):
            try:
                df = pd.read_csv(Config.ETF_LIST_CSV)
                for _, row in df.iterrows():
                    self.db_manager.upsert_etf_info(
                        ticker=row.get('Symbol', ''),
                        name=row.get('Name', ''),
                        aum=row.get('AUM', 0),
                        description=row.get('description', '')
                    )
                logger.info(f"Loaded ETF info from CSV to Supabase")
            except Exception as e:
                logger.warning(f"Error loading ETF info from CSV: {e}")
        
        # Load from pickle
        if os.path.exists(Config.ETF_INFO_PICKLE):
            try:
                with open(Config.ETF_INFO_PICKLE, 'rb') as f:
                    pickle_data = pickle.load(f)
                    for item in pickle_data:
                        if isinstance(item, dict):
                            self.db_manager.upsert_etf_info(
                                ticker=item.get('ticker', ''),
                                name=item.get('name', ''),
                                aum=item.get('aum', 0),
                                description=item.get('description', '')
                            )
                logger.info(f"Loaded ETF info from pickle to Supabase")
            except Exception as e:
                logger.warning(f"Error loading ETF info from pickle: {e}")