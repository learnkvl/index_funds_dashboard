import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# utils/charts.py - Chart Generation with Altair
import altair as alt
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

# Configure Altair
# alt.data_transformers.enable('inline')
alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('default')

class ChartGenerator:
    """Generate interactive charts using Altair"""
    
    def __init__(self):
        self.theme = Config.CHART_THEME
        
    def create_price_chart(self, df: pd.DataFrame, ticker: str) -> str:
        """Create interactive price chart"""
        try:
            if df.empty:
                return self._create_empty_chart("No data available")
            
            # Prepare data
            chart_data = df.copy()
            if 'date' not in chart_data.columns and isinstance(chart_data.index, pd.DatetimeIndex):
                chart_data = chart_data.reset_index()
                chart_data.rename(columns={'index': 'date'}, inplace=True)
            
            # Ensure date is datetime
            chart_data['date'] = pd.to_datetime(chart_data['date'])
            
            # Main price line
            price_line = alt.Chart(chart_data).mark_line(
                color='steelblue',
                strokeWidth=2
            ).encode(
                x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %Y')),
                y=alt.Y('close:Q', title='Price ($)', scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('close:Q', title='Price', format='$.2f'),
                    alt.Tooltip('volume:Q', title='Volume', format=','),
                    alt.Tooltip('rsi:Q', title='RSI', format='.2f')
                ]
            )
            
            # Add moving averages if available
            charts = [price_line]
            
            if 'sma_20' in chart_data.columns:
                sma_20 = alt.Chart(chart_data).mark_line(
                    color='orange',
                    strokeDash=[5, 5],
                    opacity=0.7
                ).encode(
                    x='date:T',
                    y='sma_20:Q',
                    tooltip=['date:T', alt.Tooltip('sma_20:Q', title='SMA 20', format='$.2f')]
                )
                charts.append(sma_20)
            
            if 'sma_50' in chart_data.columns:
                sma_50 = alt.Chart(chart_data).mark_line(
                    color='red',
                    strokeDash=[10, 5],
                    opacity=0.7
                ).encode(
                    x='date:T',
                    y='sma_50:Q',
                    tooltip=['date:T', alt.Tooltip('sma_50:Q', title='SMA 50', format='$.2f')]
                )
                charts.append(sma_50)
            
            # Combine charts
            combined_chart = alt.layer(*charts).add_selection(
                alt.selection_interval(bind='scales')
            ).properties(
                width=700,
                height=350,
                title=f'{ticker} - Price Chart with Moving Averages'
            )
            
            return combined_chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating price chart for {ticker}: {e}")
            return self._create_empty_chart("Error creating chart")
    
    def create_rsi_chart(self, df: pd.DataFrame, ticker: str) -> str:
        """Create interactive RSI chart with threshold lines"""
        try:
            if df.empty or 'rsi' not in df.columns:
                return self._create_empty_chart("No RSI data available")
            
            # Prepare data
            chart_data = df.copy()
            if 'date' not in chart_data.columns and isinstance(chart_data.index, pd.DatetimeIndex):
                chart_data = chart_data.reset_index()
                chart_data.rename(columns={'index': 'date'}, inplace=True)
            
            chart_data['date'] = pd.to_datetime(chart_data['date'])
            
            # Main RSI line
            rsi_line = alt.Chart(chart_data).mark_line(
                color='purple',
                strokeWidth=2
            ).encode(
                x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %Y')),
                y=alt.Y('rsi:Q', title='RSI', scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('rsi:Q', title='RSI', format='.2f'),
                    alt.Tooltip('close:Q', title='Price', format='$.2f')
                ]
            )
            
            # Overbought line (70)
            overbought_line = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(
                color='red',
                strokeDash=[5, 5],
                opacity=0.8
            ).encode(
                y='y:Q',
                tooltip=alt.value('Overbought Level (70)')
            )
            
            # Oversold line (30)
            oversold_line = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(
                color='green',
                strokeDash=[5, 5],
                opacity=0.8
            ).encode(
                y='y:Q',
                tooltip=alt.value('Oversold Level (30)')
            )
            
            # Highlight oversold areas
            oversold_area = alt.Chart(chart_data).mark_area(
                opacity=0.2,
                color='green'
            ).encode(
                x='date:T',
                y=alt.Y('rsi:Q', scale=alt.Scale(domain=[0, 100])),
                y2=alt.datum(0)
            ).transform_filter(
                alt.datum.rsi < 30
            )
            
            # Highlight overbought areas
            overbought_area = alt.Chart(chart_data).mark_area(
                opacity=0.2,
                color='red'
            ).encode(
                x='date:T',
                y=alt.Y('rsi:Q', scale=alt.Scale(domain=[0, 100])),
                y2=alt.datum(100)
            ).transform_filter(
                alt.datum.rsi > 70
            )
            
            # Combine all elements
            combined_chart = alt.layer(
                oversold_area,
                overbought_area,
                rsi_line,
                overbought_line,
                oversold_line
            ).add_selection(
                alt.selection_interval(bind='scales')
            ).properties(
                width=700,
                height=250,
                title=f'{ticker} - RSI Chart with Overbought/Oversold Levels'
            )
            
            return combined_chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating RSI chart for {ticker}: {e}")
            return self._create_empty_chart("Error creating RSI chart")
    
    def create_overview_chart(self, df: pd.DataFrame) -> str:
        """Create overview chart showing current RSI for all ETFs"""
        try:
            if df.empty:
                return self._create_empty_chart("No data available")
            
            # Add RSI categories for coloring
            def categorize_rsi(rsi):
                if pd.isna(rsi):
                    return 'Unknown'
                elif rsi < 30:
                    return 'Oversold'
                elif rsi > 70:
                    return 'Overbought'
                else:
                    return 'Normal'
            
            chart_data = df.copy()
            chart_data['RSI_Category'] = chart_data['RSI'].apply(categorize_rsi)
            
            # Sort by RSI for better visualization
            chart_data = chart_data.sort_values('RSI', ascending=True)
            
            # Create bar chart
            bars = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Ticker:N', title='ETF Ticker', sort=alt.SortField('RSI', order='ascending')),
                y=alt.Y('RSI:Q', title='Current RSI', scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    'RSI_Category:N',
                    title='RSI Category',
                    scale=alt.Scale(
                        domain=['Oversold', 'Normal', 'Overbought', 'Unknown'],
                        range=['#28a745', '#007bff', '#dc3545', '#6c757d']
                    )
                ),
                tooltip=[
                    'Ticker:N',
                    alt.Tooltip('RSI:Q', title='RSI', format='.2f'),
                    alt.Tooltip('Close:Q', title='Price', format='$.2f'),
                    'RSI_Category:N'
                ]
            )
            
            # Add threshold lines
            overbought_line = alt.Chart(pd.DataFrame({'threshold': [70], 'label': ['Overbought (70)']})).mark_rule(
                color='red',
                strokeDash=[3, 3],
                opacity=0.7
            ).encode(
                y='threshold:Q',
                tooltip='label:N'
            )
            
            oversold_line = alt.Chart(pd.DataFrame({'threshold': [30], 'label': ['Oversold (30)']})).mark_rule(
                color='green',
                strokeDash=[3, 3],
                opacity=0.7
            ).encode(
                y='threshold:Q',
                tooltip='label:N'
            )
            
            # Combine chart elements
            combined_chart = alt.layer(
                bars,
                overbought_line,
                oversold_line
            ).properties(
                width=900,
                height=400,
                title='Current RSI Overview - All ETFs'
            )
            
            return combined_chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating overview chart: {e}")
            return self._create_empty_chart("Error creating overview chart")
    
    def create_volume_chart(self, df: pd.DataFrame, ticker: str) -> str:
        """Create volume chart"""
        try:
            if df.empty or 'volume' not in df.columns:
                return self._create_empty_chart("No volume data available")
            
            chart_data = df.copy()
            if 'date' not in chart_data.columns and isinstance(chart_data.index, pd.DatetimeIndex):
                chart_data = chart_data.reset_index()
                chart_data.rename(columns={'index': 'date'}, inplace=True)
            
            chart_data['date'] = pd.to_datetime(chart_data['date'])
            
            # Volume bars
            volume_bars = alt.Chart(chart_data).mark_bar(
                opacity=0.7,
                color='lightblue'
            ).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('volume:Q', title='Volume'),
                tooltip=[
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('volume:Q', title='Volume', format=','),
                    alt.Tooltip('close:Q', title='Price', format='$.2f')
                ]
            )
            
            # Add volume moving average if available
            if 'volume_sma' in chart_data.columns:
                volume_sma = alt.Chart(chart_data).mark_line(
                    color='red',
                    strokeWidth=2
                ).encode(
                    x='date:T',
                    y='volume_sma:Q',
                    tooltip=[
                        alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                        alt.Tooltip('volume_sma:Q', title='Volume SMA', format=',')
                    ]
                )
                
                combined_chart = alt.layer(volume_bars, volume_sma)
            else:
                combined_chart = volume_bars
            
            final_chart = combined_chart.add_selection(
                alt.selection_interval(bind='scales')
            ).properties(
                width=700,
                height=200,
                title=f'{ticker} - Volume Chart'
            )
            
            return final_chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating volume chart for {ticker}: {e}")
            return self._create_empty_chart("Error creating volume chart")
    
    def create_macd_chart(self, df: pd.DataFrame, ticker: str) -> str:
        """Create MACD chart"""
        try:
            if df.empty or 'macd' not in df.columns:
                return self._create_empty_chart("No MACD data available")
            
            chart_data = df.copy()
            if 'date' not in chart_data.columns and isinstance(chart_data.index, pd.DatetimeIndex):
                chart_data = chart_data.reset_index()
                chart_data.rename(columns={'index': 'date'}, inplace=True)
            
            chart_data['date'] = pd.to_datetime(chart_data['date'])
            
            # MACD line
            macd_line = alt.Chart(chart_data).mark_line(
                color='blue',
                strokeWidth=2
            ).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('macd:Q', title='MACD'),
                tooltip=[
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('macd:Q', title='MACD', format='.4f')
                ]
            )
            
            charts = [macd_line]
            
            # Signal line
            if 'macd_signal' in chart_data.columns:
                signal_line = alt.Chart(chart_data).mark_line(
                    color='red',
                    strokeWidth=1,
                    strokeDash=[5, 5]
                ).encode(
                    x='date:T',
                    y='macd_signal:Q',
                    tooltip=[
                        alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                        alt.Tooltip('macd_signal:Q', title='Signal', format='.4f')
                    ]
                )
                charts.append(signal_line)
            
            # Histogram
            if 'macd_histogram' in chart_data.columns:
                histogram = alt.Chart(chart_data).mark_bar(
                    opacity=0.6,
                    color='gray'
                ).encode(
                    x='date:T',
                    y='macd_histogram:Q',
                    tooltip=[
                        alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                        alt.Tooltip('macd_histogram:Q', title='Histogram', format='.4f')
                    ]
                )
                charts.append(histogram)
            
            # Zero line
            zero_line = alt.Chart(pd.DataFrame({'zero': [0]})).mark_rule(
                color='black',
                strokeDash=[2, 2],
                opacity=0.5
            ).encode(y='zero:Q')
            charts.append(zero_line)
            
            combined_chart = alt.layer(*charts).add_selection(
                alt.selection_interval(bind='scales')
            ).properties(
                width=700,
                height=200,
                title=f'{ticker} - MACD Chart'
            )
            
            return combined_chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating MACD chart for {ticker}: {e}")
            return self._create_empty_chart("Error creating MACD chart")
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        try:
            if correlation_matrix.empty:
                return self._create_empty_chart("No correlation data available")
            
            # Reshape correlation matrix for Altair
            corr_data = correlation_matrix.reset_index()
            corr_melted = corr_data.melt(id_vars='index', var_name='ticker2', value_name='correlation')
            corr_melted.rename(columns={'index': 'ticker1'}, inplace=True)
            
            heatmap = alt.Chart(corr_melted).mark_rect().encode(
                x=alt.X('ticker1:O', title='ETF'),
                y=alt.Y('ticker2:O', title='ETF'),
                color=alt.Color(
                    'correlation:Q',
                    title='Correlation',
                    scale=alt.Scale(scheme='redblue', domain=[-1, 1])
                ),
                tooltip=[
                    'ticker1:O',
                    'ticker2:O',
                    alt.Tooltip('correlation:Q', title='Correlation', format='.3f')
                ]
            ).properties(
                width=500,
                height=500,
                title='ETF Correlation Heatmap'
            )
            
            return heatmap.to_json()
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return self._create_empty_chart("Error creating correlation heatmap")
    
    def create_candlestick_chart(self, df: pd.DataFrame, ticker: str) -> str:
        """Create candlestick chart"""
        try:
            if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                return self._create_empty_chart("Insufficient OHLC data for candlestick chart")
            
            chart_data = df.copy()
            if 'date' not in chart_data.columns and isinstance(chart_data.index, pd.DatetimeIndex):
                chart_data = chart_data.reset_index()
                chart_data.rename(columns={'index': 'date'}, inplace=True)
            
            chart_data['date'] = pd.to_datetime(chart_data['date'])
            
            # Add color for up/down days
            chart_data['color'] = np.where(chart_data['close'] >= chart_data['open'], 'green', 'red')
            
            # High-Low lines
            hl_lines = alt.Chart(chart_data).mark_rule().encode(
                x=alt.X('date:T', title='Date'),
                y='low:Q',
                y2='high:Q',
                color=alt.Color('color:N', scale=alt.Scale(range=['red', 'green']), legend=None),
                tooltip=[
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('open:Q', title='Open', format='$.2f'),
                    alt.Tooltip('high:Q', title='High', format='$.2f'),
                    alt.Tooltip('low:Q', title='Low', format='$.2f'),
                    alt.Tooltip('close:Q', title='Close', format='$.2f'),
                    alt.Tooltip('volume:Q', title='Volume', format=',')
                ]
            )
            
            # Open-Close bars
            oc_bars = alt.Chart(chart_data).mark_bar(size=5).encode(
                x='date:T',
                y='open:Q',
                y2='close:Q',
                color=alt.Color('color:N', scale=alt.Scale(range=['red', 'green']), legend=None),
                tooltip=[
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('open:Q', title='Open', format='$.2f'),
                    alt.Tooltip('high:Q', title='High', format='$.2f'),
                    alt.Tooltip('low:Q', title='Low', format='$.2f'),
                    alt.Tooltip('close:Q', title='Close', format='$.2f'),
                    alt.Tooltip('volume:Q', title='Volume', format=',')
                ]
            )
            
            combined_chart = alt.layer(hl_lines, oc_bars).add_selection(
                alt.selection_interval(bind='scales')
            ).properties(
                width=700,
                height=400,
                title=f'{ticker} - Candlestick Chart'
            )
            
            return combined_chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart for {ticker}: {e}")
            return self._create_empty_chart("Error creating candlestick chart")
    
    def create_multi_timeframe_chart(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, 
                                   ticker: str) -> str:
        """Create multi-timeframe chart comparing daily and weekly data"""
        try:
            if daily_df.empty and weekly_df.empty:
                return self._create_empty_chart("No data available for multi-timeframe chart")
            
            charts = []
            
            # Daily chart
            if not daily_df.empty:
                daily_data = daily_df.copy()
                if 'date' not in daily_data.columns:
                    daily_data = daily_data.reset_index()
                    daily_data.rename(columns={'index': 'date'}, inplace=True)
                daily_data['timeframe'] = 'Daily'
                
                daily_chart = alt.Chart(daily_data).mark_line(
                    color='blue',
                    opacity=0.7
                ).encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('close:Q', title='Price ($)'),
                    color=alt.value('blue'),
                    tooltip=['date:T', 'close:Q', 'timeframe:N']
                )
                charts.append(daily_chart)
            
            # Weekly chart
            if not weekly_df.empty:
                weekly_data = weekly_df.copy()
                if 'date' not in weekly_data.columns:
                    weekly_data = weekly_data.reset_index()
                    weekly_data.rename(columns={'index': 'date'}, inplace=True)
                weekly_data['timeframe'] = 'Weekly'
                
                weekly_chart = alt.Chart(weekly_data).mark_line(
                    color='red',
                    strokeWidth=2
                ).encode(
                    x='date:T',
                    y='close:Q',
                    color=alt.value('red'),
                    tooltip=['date:T', 'close:Q', 'timeframe:N']
                )
                charts.append(weekly_chart)
            
            if charts:
                combined_chart = alt.layer(*charts).add_selection(
                    alt.selection_interval(bind='scales')
                ).properties(
                    width=700,
                    height=400,
                    title=f'{ticker} - Multi-Timeframe Chart (Daily vs Weekly)'
                )
                
                return combined_chart.to_json()
            else:
                return self._create_empty_chart("No valid data for multi-timeframe chart")
            
        except Exception as e:
            logger.error(f"Error creating multi-timeframe chart for {ticker}: {e}")
            return self._create_empty_chart("Error creating multi-timeframe chart")
    
    def create_performance_comparison_chart(self, performance_data: pd.DataFrame) -> str:
        """Create performance comparison chart for multiple ETFs"""
        try:
            if performance_data.empty:
                return self._create_empty_chart("No performance data available")
            
            # Normalize data to show percentage returns from start
            normalized_data = performance_data.copy()
            for col in normalized_data.columns:
                if col != 'date':
                    first_value = normalized_data[col].iloc[0]
                    normalized_data[col] = ((normalized_data[col] / first_value) - 1) * 100
            
            # Melt data for Altair
            melted_data = normalized_data.melt(
                id_vars='date',
                var_name='ticker',
                value_name='return_pct'
            )
            
            chart = alt.Chart(melted_data).mark_line().encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('return_pct:Q', title='Return (%)'),
                color=alt.Color('ticker:N', title='ETF'),
                tooltip=[
                    'ticker:N',
                    alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('return_pct:Q', title='Return (%)', format='.2f')
                ]
            ).add_selection(
                alt.selection_interval(bind='scales')
            ).properties(
                width=700,
                height=400,
                title='ETF Performance Comparison (Normalized Returns)'
            )
            
            return chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating performance comparison chart: {e}")
            return self._create_empty_chart("Error creating performance comparison chart")
    
    def _create_empty_chart(self, message: str) -> str:
        """Create an empty chart with a message"""
        try:
            empty_chart = alt.Chart(pd.DataFrame({
                'x': [0],
                'y': [0],
                'message': [message]
            })).mark_text(
                fontSize=16,
                color='gray'
            ).encode(
                x=alt.X('x:Q', axis=None),
                y=alt.Y('y:Q', axis=None),
                text='message:N'
            ).properties(
                width=400,
                height=200,
                title='Chart Not Available'
            )
            
            return empty_chart.to_json()
            
        except Exception as e:
            logger.error(f"Error creating empty chart: {e}")
            return '{"error": "Chart generation failed"}'
    
    def create_dashboard_summary_chart(self, summary_data: pd.DataFrame) -> str:
        """Create a summary chart for the dashboard"""
        try:
            if summary_data.empty:
                return self._create_empty_chart("No summary data available")
            
            # Create a small multiples chart showing key metrics
            base = alt.Chart(summary_data).add_selection(
                alt.selection_interval()
            ).properties(
                width=150,
                height=100
            )
            
            # RSI distribution
            rsi_hist = base.mark_bar().encode(
                x=alt.X('RSI:Q', bin=alt.Bin(maxbins=10), title='RSI'),
                y=alt.Y('count():Q', title='Count'),
                color=alt.condition(
                    alt.datum.RSI < 30,
                    alt.value('green'),
                    alt.condition(
                        alt.datum.RSI > 70,
                        alt.value('red'),
                        alt.value('blue')
                    )
                )
            ).properties(
                title='RSI Distribution'
            )
            
            # Price distribution
            price_hist = base.mark_bar(color='steelblue').encode(
                x=alt.X('Close:Q', bin=alt.Bin(maxbins=10), title='Price ($)'),
                y=alt.Y('count():Q', title='Count')
            ).properties(
                title='Price Distribution'
            )
            
            # Volume vs RSI scatter
            volume_rsi = base.mark_circle(size=60).encode(
                x=alt.X('Volume:Q', title='Volume', scale=alt.Scale(type='log')),
                y=alt.Y('RSI:Q', title='RSI'),
                color=alt.Color('RSI:Q', scale=alt.Scale(scheme='redblue')),
                tooltip=['Ticker:N', 'RSI:Q', 'Volume:Q', 'Close:Q']
            ).properties(
                title='Volume vs RSI'
            )
            
            # Combine charts
            combined = alt.hconcat(
                rsi_hist,
                price_hist,
                volume_rsi
            ).properties(
                title='ETF Dashboard Summary'
            )
            
            return combined.to_json()
            
        except Exception as e:
            logger.error(f"Error creating dashboard summary chart: {e}")
            return self._create_empty_chart("Error creating dashboard summary")