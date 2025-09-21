# app.py - Complete Enhanced Flask Application for ETF Dashboard with Supabase PostgreSQL
from flask import Flask, render_template, request, jsonify
from services.etf_service import ETFService
from database.db_manager import DatabaseManager
from config import Config
import logging
import pandas as pd
import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # === SUPABASE INITIALIZATION ===
    try:
        # Initialize database manager for Supabase PostgreSQL
        # No need to pass DATABASE_PATH - it gets DATABASE_URL from environment
        db_manager = DatabaseManager()
        
        # Verify Supabase connection on startup
        if not db_manager.check_connection():
            logger.error("Failed to connect to Supabase PostgreSQL database")
            logger.error("Please check your DATABASE_URL environment variable")
            raise ConnectionError("Cannot connect to Supabase database")
        
        logger.info("Successfully connected to Supabase PostgreSQL database")
        
    except ValueError as e:
        logger.error(f"Supabase configuration error: {e}")
        logger.error("Please ensure DATABASE_URL is set in your environment variables")
        logger.error("Example: DATABASE_URL=postgresql://postgres:password@db.project.supabase.co:5432/postgres")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize Supabase database connection: {e}")
        raise
    
    # Initialize ETF service with Supabase-backed database
    try:
        etf_service = ETFService(db_manager)
        logger.info("ETFService initialized successfully with Supabase backend")
    except Exception as e:
        logger.error(f"Failed to initialize ETFService: {e}")
        raise
    
    # ============= WEB ROUTES (HTML Pages) =============
    
    @app.route('/')
    def dashboard():
        """Main dashboard page"""
        try:
            etfs = etf_service.get_available_etfs()
            logger.info(f"Dashboard loaded with {len(etfs)} available ETFs")
            return render_template('dashboard.html', etfs=etfs)
        except Exception as e:
            logger.error(f"Error loading dashboard: {e}")
            return render_template('error.html', error="Failed to load dashboard"), 500

    @app.route('/etf/<ticker>')
    def etf_detail(ticker):
        """Individual ETF detail page"""
        logger.info(f"ETF detail page requested for {ticker}")
        return render_template('etf_detail.html', ticker=ticker)

    @app.route('/screener')
    def etf_screener():
        return render_template('etf_screener.html')

    @app.route('/settings')
    def settings():
        return render_template('settings.html')

    @app.route('/compare')
    def compare():
        return render_template('compare.html')

    @app.route('/portfolio')
    def portfolio():
        return render_template('portfolio.html')

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/help')
    def help():
        return render_template('help.html')

    # ============= CORE API ENDPOINTS =============

    @app.route('/api/overview')
    def api_overview():
        """API endpoint for overview data"""
        try:
            period = request.args.get('period', '6mo')
            threshold = float(request.args.get('threshold', 30))
            
            logger.info(f"Overview API called with period={period}, threshold={threshold}")
            
            # Get overview data from Supabase
            overview_data = etf_service.get_overview_data(period=period, threshold=threshold)
            
            if overview_data is None:
                logger.warning("No overview data available from Supabase")
                return jsonify({'error': 'No data available'}), 500
            
            logger.info(f"Overview data retrieved: {overview_data.get('total_etfs', 0)} ETFs processed")
            return jsonify(overview_data)
            
        except Exception as e:
            logger.error(f"Error in overview API: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/etf/<ticker>')
    def api_etf_detail(ticker):
        """Enhanced API endpoint for individual ETF data from Supabase"""
        try:
            period = request.args.get('period', '6mo')
            include_signals = request.args.get('include_signals', 'true').lower() == 'true'
            
            logger.info(f"ETF detail API called for {ticker} with period={period}")
            
            etf_data = etf_service.get_etf_detail(ticker, period=period)
            
            if etf_data is None:
                logger.warning(f"No data available for {ticker} in Supabase")
                return jsonify({'error': f'No data available for {ticker}'}), 404

            # Add enhanced signal information if requested
            if include_signals:
                try:
                    # Get the latest signal data for this specific ETF
                    signals = etf_service.get_all_etf_signals(period=period)
                    etf_signal = next((s for s in signals if s['ticker'] == ticker), None)
                    
                    if etf_signal:
                        etf_data['enhanced_signal'] = {
                            'signal': etf_signal.get('signal'),
                            'signal_strength': etf_signal.get('signal_strength'),
                            'signal_components': etf_signal.get('signal_components', {}),
                            'ema_50': etf_signal.get('ema_50'),
                            'ema_200': etf_signal.get('ema_200'),
                            'macd': etf_signal.get('macd'),
                            'macd_signal': etf_signal.get('macd_signal')
                        }
                        logger.info(f"Enhanced signals added for {ticker}")
                        
                except Exception as signal_error:
                    logger.warning(f"Could not get enhanced signals for {ticker}: {signal_error}")

            # Ensure chart JSON is always valid
            def empty_chart_json(message):
                import altair as alt
                import pandas as pd
                empty_chart = alt.Chart(pd.DataFrame({'x': [0], 'y': [0], 'message': [message]})).mark_text(
                    fontSize=16, color='gray'
                ).encode(
                    x=alt.X('x:Q', axis=None),
                    y=alt.Y('y:Q', axis=None),
                    text='message:N'
                ).properties(
                    width=400, height=200, title='Chart Not Available'
                )
                return empty_chart.to_json()

            if not etf_data.get('price_chart'):
                etf_data['price_chart'] = empty_chart_json("No price data available")
            if not etf_data.get('rsi_chart'):
                etf_data['rsi_chart'] = empty_chart_json("No RSI data available")

            logger.info(f"ETF API response for {ticker} with {len(etf_data.keys())} data fields")
            return jsonify(etf_data)

        except Exception as e:
            logger.error(f"Error in ETF detail API for {ticker}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/refresh/<ticker>')
    def api_refresh_etf(ticker):
        """Force refresh ETF data from external source to Supabase"""
        try:
            logger.info(f"Force refresh requested for {ticker}")
            success = etf_service.force_refresh_etf(ticker)
            if success:
                logger.info(f"Successfully refreshed {ticker} in Supabase")
                return jsonify({'message': f'Successfully refreshed {ticker}'}), 200
            else:
                logger.warning(f"Failed to refresh {ticker}")
                return jsonify({'error': f'Failed to refresh {ticker}'}), 500
        except Exception as e:
            logger.error(f"Error refreshing {ticker}: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/health')
    def api_health():
        """Health check endpoint for Supabase system"""
        try:
            db_status = db_manager.check_connection()
            etf_count = len(etf_service.get_available_etfs())
            
            # Additional Supabase-specific health checks
            health_info = {
                'status': 'healthy',
                'database_type': 'Supabase PostgreSQL',
                'database': 'connected' if db_status else 'disconnected',
                'etf_count': etf_count,
                'enhanced_features': True,
                'version': '2.0_supabase',
                'connection_string_configured': bool(os.getenv('DATABASE_URL')),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Test a simple query to Supabase
            if db_status:
                try:
                    test_info = db_manager.get_etf_info()
                    health_info['etf_info_records'] = len(test_info) if not test_info.empty else 0
                    health_info['database_query_test'] = 'passed'
                except Exception as e:
                    health_info['database_query_test'] = f'failed: {str(e)}'
                    health_info['status'] = 'degraded'
            
            return jsonify(health_info)
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'database_type': 'Supabase PostgreSQL',
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }), 500

    # ============= ENHANCED SIGNAL API ENDPOINTS =============

    @app.route('/api/etfs/signals')
    def api_etfs_signals():
        """Enhanced API endpoint for all ETF signals from Supabase with better error handling"""
        try:
            period = request.args.get('period', '6mo')
            include_components = request.args.get('include_components', 'false').lower() == 'true'
            
            logger.info(f"ETF signals request received for period: {period}")
            
            # Get available ETFs
            available_etfs = etf_service.get_available_etfs()
            logger.info(f"Available ETFs count: {len(available_etfs)}")
            
            if not available_etfs:
                logger.warning("No ETFs available in service")
                return jsonify({'error': 'No ETFs available', 'etfs': []})
            
            # Call the enhanced service method
            logger.info("Calling enhanced etf_service.get_all_etf_signals()...")
            etf_signals = etf_service.get_all_etf_signals(period=period)
            
            logger.info(f"ETF signals returned: {len(etf_signals)} items")
            
            if not etf_signals:
                logger.warning("No ETF signals returned from enhanced service")
                return jsonify([])
            
            # Optionally filter out signal components for lighter payload
            if not include_components:
                for signal in etf_signals:
                    signal.pop('signal_components', None)
            
            # Log summary statistics
            signal_counts = {}
            for signal in etf_signals:
                sig_type = signal.get('signal', 'UNKNOWN')
                signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1
            
            logger.info(f"Signal distribution: {signal_counts}")
            
            return jsonify(etf_signals)
            
        except Exception as e:
            logger.error(f"Error in enhanced ETF signals API: {str(e)}", exc_info=True)
            return jsonify({
                'error': f'Enhanced ETF signals processing failed: {str(e)}',
                'debug_info': {
                    'available_etfs': len(etf_service.get_available_etfs()) if hasattr(etf_service, 'get_available_etfs') else 'unknown',
                    'service_initialized': etf_service is not None,
                    'database_type': 'Supabase PostgreSQL',
                    'timestamp': datetime.datetime.now().isoformat()
                }
            }), 500

    @app.route('/api/etfs/signals/summary')
    def api_etfs_signals_summary():
        """API endpoint for enhanced signal summary statistics from Supabase"""
        try:
            period = request.args.get('period', '6mo')
            logger.info(f"ETF signals summary request for period: {period}")
            
            # Get all signals
            etf_signals = etf_service.get_all_etf_signals(period=period)
            
            if not etf_signals:
                return jsonify({'error': 'No ETF signals available'})
            
            # Calculate enhanced summary statistics
            summary = {
                'total_etfs': len(etf_signals),
                'strong_buy': len([etf for etf in etf_signals if etf.get('signal') == 'STRONG_BUY']),
                'buy': len([etf for etf in etf_signals if etf.get('signal') == 'BUY']),
                'hold': len([etf for etf in etf_signals if etf.get('signal') == 'HOLD']),
                'sell': len([etf for etf in etf_signals if etf.get('signal') == 'SELL']),
                'strong_sell': len([etf for etf in etf_signals if etf.get('signal') == 'STRONG_SELL']),
                'error_signals': len([etf for etf in etf_signals if etf.get('signal') == 'ERROR']),
                
                # Enhanced trend analysis
                'golden_cross': len([etf for etf in etf_signals if etf.get('ema_50') and etf.get('ema_200') and etf.get('ema_50') > etf.get('ema_200')]),
                'death_cross': len([etf for etf in etf_signals if etf.get('ema_50') and etf.get('ema_200') and etf.get('ema_50') < etf.get('ema_200')]),
                
                # MACD analysis
                'macd_bullish': len([etf for etf in etf_signals if etf.get('macd') and etf.get('macd_signal') and etf.get('macd') > etf.get('macd_signal')]),
                'macd_bearish': len([etf for etf in etf_signals if etf.get('macd') and etf.get('macd_signal') and etf.get('macd') < etf.get('macd_signal')]),
                
                # Signal strength analysis
                'strong_signals': len([etf for etf in etf_signals if etf.get('signal_strength') == 'STRONG']),
                'moderate_signals': len([etf for etf in etf_signals if etf.get('signal_strength') == 'MODERATE']),
                'weak_signals': len([etf for etf in etf_signals if etf.get('signal_strength') == 'WEAK']),
                
                # RSI analysis
                'rsi_oversold': len([etf for etf in etf_signals if etf.get('rsi') and etf.get('rsi') < 30]),
                'rsi_overbought': len([etf for etf in etf_signals if etf.get('rsi') and etf.get('rsi') > 70]),
                'rsi_ideal_zone': len([etf for etf in etf_signals if etf.get('rsi') and 40 <= etf.get('rsi') <= 60]),
                
                'database_type': 'Supabase PostgreSQL',
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Signal summary generated: {summary['total_etfs']} total ETFs")
            return jsonify(summary)
            
        except Exception as e:
            logger.error(f"Error in ETF signals summary API: {str(e)}")
            return jsonify({'error': f'Summary processing failed: {str(e)}'}), 500

    @app.route('/api/etfs/signals/filtered')
    def api_etfs_signals_filtered():
        """API endpoint for filtered signals based on enhanced criteria from Supabase"""
        try:
            # Get filter parameters
            signal_type = request.args.get('signal_type')  # STRONG_BUY, BUY, etc.
            signal_strength = request.args.get('signal_strength')  # STRONG, MODERATE, WEAK
            trend_type = request.args.get('trend_type')  # golden_cross, death_cross
            rsi_min = request.args.get('rsi_min', type=float)
            rsi_max = request.args.get('rsi_max', type=float)
            macd_status = request.args.get('macd_status')  # bullish, bearish
            adx_min = request.args.get('adx_min', type=float)
            period = request.args.get('period', '6mo')
            
            logger.info(f"Filtered signals request: signal_type={signal_type}, trend_type={trend_type}, period={period}")
            
            # Get all signals from Supabase
            etf_signals = etf_service.get_all_etf_signals(period=period)
            
            # Apply filters
            filtered_signals = []
            for etf in etf_signals:
                # Signal type filter
                if signal_type and etf.get('signal') != signal_type:
                    continue
                    
                # Signal strength filter
                if signal_strength and etf.get('signal_strength') != signal_strength:
                    continue
                    
                # Trend type filter
                if trend_type:
                    ema_50 = etf.get('ema_50')
                    ema_200 = etf.get('ema_200')
                    if trend_type == 'golden_cross' and not (ema_50 and ema_200 and ema_50 > ema_200):
                        continue
                    elif trend_type == 'death_cross' and not (ema_50 and ema_200 and ema_50 < ema_200):
                        continue
                
                # RSI filters
                rsi = etf.get('rsi')
                if rsi:
                    if rsi_min is not None and rsi < rsi_min:
                        continue
                    if rsi_max is not None and rsi > rsi_max:
                        continue
                
                # MACD status filter
                if macd_status:
                    macd = etf.get('macd')
                    macd_signal_line = etf.get('macd_signal')
                    if macd and macd_signal_line:
                        if macd_status == 'bullish' and macd <= macd_signal_line:
                            continue
                        elif macd_status == 'bearish' and macd >= macd_signal_line:
                            continue
                
                # ADX filter
                if adx_min is not None:
                    adx = etf.get('adx')
                    if not adx or adx < adx_min:
                        continue
                
                filtered_signals.append(etf)
            
            logger.info(f"Filtered signals: {len(filtered_signals)} ETFs match criteria")
            
            return jsonify({
                'signals': filtered_signals,
                'total_count': len(filtered_signals),
                'filters_applied': {
                    'signal_type': signal_type,
                    'signal_strength': signal_strength,
                    'trend_type': trend_type,
                    'rsi_min': rsi_min,
                    'rsi_max': rsi_max,
                    'macd_status': macd_status,
                    'adx_min': adx_min,
                    'period': period
                },
                'database_type': 'Supabase PostgreSQL'
            })
            
        except Exception as e:
            logger.error(f"Error in filtered signals API: {str(e)}")
            return jsonify({'error': f'Filtered signals processing failed: {str(e)}'}), 500

    # Health check specifically for enhanced signals
    @app.route('/api/etfs/signals/health')
    def api_etfs_signals_health():
        """Health check for enhanced ETF signals functionality with Supabase"""
        try:
            health_info = {
                'service_available': etf_service is not None,
                'etf_count': len(etf_service.get_available_etfs()) if etf_service else 0,
                'database_connected': db_manager.check_connection() if db_manager else False,
                'database_type': 'Supabase PostgreSQL',
                'indicators_available': hasattr(etf_service, 'indicators'),
                'enhanced_features': True,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Test a single ETF
            if etf_service and len(etf_service.get_available_etfs()) > 0:
                test_ticker = etf_service.get_available_etfs()[0]
                try:
                    test_data = etf_service.get_etf_data_smart(test_ticker, '1mo')
                    health_info['test_data_available'] = not test_data.empty
                    health_info['test_ticker'] = test_ticker
                    if not test_data.empty:
                        health_info['test_data_columns'] = list(test_data.columns)
                        health_info['test_data_rows'] = len(test_data)
                        
                        # Check for enhanced indicators
                        enhanced_indicators = ['ema_50', 'ema_200', 'macd', 'macd_signal', 'supertrend', 'adx']
                        available_enhanced = [ind for ind in enhanced_indicators if ind in test_data.columns]
                        health_info['enhanced_indicators_available'] = available_enhanced
                        health_info['enhanced_indicators_count'] = len(available_enhanced)
                        
                except Exception as e:
                    health_info['test_data_error'] = str(e)
            
            return jsonify(health_info)
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'database_type': 'Supabase PostgreSQL',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500

    # ============= FILTERING API ENDPOINTS =============

    @app.route('/api/etfs/filter')
    def api_etfs_filter():
        """Enhanced API endpoint for filtering ETFs based on criteria from Supabase"""
        try:
            # Get filter parameters from query string
            rsi_min = request.args.get('rsi_min', type=float)
            rsi_max = request.args.get('rsi_max', type=float)
            price_min = request.args.get('price_min', type=float)
            price_max = request.args.get('price_max', type=float)
            volume_min = request.args.get('volume_min', type=int)
            period = request.args.get('period', '6mo')
            
            # Enhanced filters
            signal_type = request.args.get('signal_type')
            trend_type = request.args.get('trend_type')  # golden_cross, death_cross
            adx_min = request.args.get('adx_min', type=float)
            
            # Log the filtering request
            logger.info(f"ETF filtering request: RSI({rsi_min}-{rsi_max}), Price({price_min}-{price_max}), Volume({volume_min}), Signal({signal_type}), Trend({trend_type}), Period({period})")
            
            # Use the ETF service filtering method
            filtered_results = etf_service.get_etfs_by_criteria(
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                volume_min=volume_min,
                price_min=price_min,
                price_max=price_max,
                period=period
            )
            
            # Apply additional enhanced filters if needed
            if signal_type or trend_type or adx_min is not None:
                enhanced_filtered = []
                for etf in filtered_results:
                    try:
                        # Get enhanced signal data
                        ticker = etf['ticker']
                        signals = etf_service.get_all_etf_signals(period=period)
                        etf_signal = next((s for s in signals if s['ticker'] == ticker), None)
                        
                        if etf_signal:
                            # Apply enhanced filters
                            if signal_type and etf_signal.get('signal') != signal_type:
                                continue
                            
                            if trend_type:
                                ema_50 = etf_signal.get('ema_50')
                                ema_200 = etf_signal.get('ema_200')
                                if trend_type == 'golden_cross' and not (ema_50 and ema_200 and ema_50 > ema_200):
                                    continue
                                elif trend_type == 'death_cross' and not (ema_50 and ema_200 and ema_50 < ema_200):
                                    continue
                            
                            if adx_min is not None:
                                adx = etf_signal.get('adx')
                                if not adx or adx < adx_min:
                                    continue
                            
                            # Add enhanced data to result
                            etf.update({
                                'signal': etf_signal.get('signal', 'HOLD'),
                                'signal_strength': etf_signal.get('signal_strength', 'WEAK'),
                                'ema_50': etf_signal.get('ema_50'),
                                'ema_200': etf_signal.get('ema_200'),
                                'adx': etf_signal.get('adx'),
                                'trend': 'Golden Cross' if (etf_signal.get('ema_50') and etf_signal.get('ema_200') and etf_signal.get('ema_50') > etf_signal.get('ema_200')) else 'Death Cross' if (etf_signal.get('ema_50') and etf_signal.get('ema_200') and etf_signal.get('ema_50') < etf_signal.get('ema_200')) else 'Neutral'
                            })
                        
                        enhanced_filtered.append(etf)
                        
                    except Exception as e:
                        logger.warning(f"Error applying enhanced filters to {etf.get('ticker', 'unknown')}: {e}")
                        continue
                
                filtered_results = enhanced_filtered
            
            logger.info(f"ETF filtering completed: {len(filtered_results)} ETFs match criteria")
            return jsonify(filtered_results)
            
        except Exception as e:
            logger.error(f"Error in enhanced ETF filtering API: {str(e)}")
            return jsonify({'error': f'Enhanced filter processing failed: {str(e)}'}), 500

    # Helper endpoint to get available filter options
    @app.route('/api/etfs/filter/options')
    def api_etfs_filter_options():
        """Get enhanced available filter options and ranges from Supabase data"""
        try:
            # Get a sample of ETFs to determine ranges
            available_etfs = etf_service.get_available_etfs()
            sample_size = min(50, len(available_etfs))  # Sample for performance
            sample_etfs = available_etfs[:sample_size]
            
            price_range = {'min': float('inf'), 'max': 0}
            rsi_range = {'min': 100, 'max': 0}
            volume_range = {'min': float('inf'), 'max': 0}
            adx_range = {'min': 100, 'max': 0}
            
            valid_samples = 0
            
            for ticker in sample_etfs:
                try:
                    data = etf_service.get_etf_data_smart(ticker, '1mo')  # Use shorter period for speed
                    
                    if data.empty:
                        continue
                    
                    latest = data.iloc[-1]
                    current_rsi = latest.get('rsi', None)
                    current_price = latest.get('close', None)
                    current_volume = latest.get('volume', None)
                    current_adx = latest.get('adx', None)
                    
                    if current_rsi is not None and current_price is not None and current_volume is not None:
                        price_range['min'] = min(price_range['min'], current_price)
                        price_range['max'] = max(price_range['max'], current_price)
                        rsi_range['min'] = min(rsi_range['min'], current_rsi)
                        rsi_range['max'] = max(rsi_range['max'], current_rsi)
                        volume_range['min'] = min(volume_range['min'], current_volume)
                        volume_range['max'] = max(volume_range['max'], current_volume)
                        
                        if current_adx is not None:
                            adx_range['min'] = min(adx_range['min'], current_adx)
                            adx_range['max'] = max(adx_range['max'], current_adx)
                        
                        valid_samples += 1
                    
                except Exception as e:
                    logger.warning(f"Error sampling {ticker}: {e}")
                    continue
            
            # Provide reasonable defaults if sampling failed
            if valid_samples == 0:
                price_range = {'min': 10.0, 'max': 500.0}
                rsi_range = {'min': 0.0, 'max': 100.0}
                volume_range = {'min': 100000, 'max': 100000000}
                adx_range = {'min': 0.0, 'max': 100.0}
            
            # Round values for better UX
            price_range['min'] = round(price_range['min'], 2)
            price_range['max'] = round(price_range['max'], 2)
            rsi_range['min'] = round(rsi_range['min'], 1)
            rsi_range['max'] = round(rsi_range['max'], 1)
            adx_range['min'] = round(adx_range['min'], 1)
            adx_range['max'] = round(adx_range['max'], 1)
            
            filter_options = {
                'price_range': price_range,
                'rsi_range': rsi_range,
                'volume_range': volume_range,
                'adx_range': adx_range,
                'available_periods': ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                'signal_types': ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'],
                'signal_strengths': ['STRONG', 'MODERATE', 'WEAK'],
                'trend_types': ['golden_cross', 'death_cross'],
                'total_etfs_available': len(available_etfs),
                'sample_size': valid_samples,
                'database_type': 'Supabase PostgreSQL',
                'suggested_filters': {
                    'strong_buy_signals': {'signal_type': 'STRONG_BUY', 'description': 'ETFs with strong buy signals'},
                    'golden_cross_etfs': {'trend_type': 'golden_cross', 'description': 'ETFs in golden cross (EMA 50 > EMA 200)'},
                    'oversold_strong_trend': {'rsi_max': 30, 'adx_min': 25, 'description': 'Oversold ETFs with strong trend'},
                    'ideal_rsi_zone': {'rsi_min': 40, 'rsi_max': 60, 'description': 'ETFs in ideal RSI zone (40-60)'},
                    'high_volume': {'volume_min': 1000000, 'description': 'ETFs with volume above 1M'},
                    'low_price': {'price_max': 50, 'description': 'ETFs priced under $50'},
                    'mid_price': {'price_min': 50, 'price_max': 200, 'description': 'ETFs priced $50-$200'}
                }
            }
            
            logger.info(f"Filter options generated from {valid_samples} ETF samples")
            return jsonify(filter_options)
            
        except Exception as e:
            logger.error(f"Error getting enhanced filter options: {str(e)}")
            return jsonify({'error': f'Failed to get enhanced filter options: {str(e)}'}), 500

    # ============= CONFIGURATION ENDPOINT =============

    @app.route('/api/config/indicators')
    def api_config_indicators():
        """Get configuration information about available enhanced indicators"""
        try:
            config_info = {
                'database_type': 'Supabase PostgreSQL',
                'available_indicators': [
                    'rsi', 'ema_50', 'ema_200', 'sma_20', 'sma_50', 'macd', 'macd_signal', 
                    'macd_histogram', 'supertrend', 'adx', 'volume_sma', 'parabolic_sar',
                    'tenkan_sen', 'kijun_sen', 'bb_upper', 'bb_middle', 'bb_lower'
                ],
                'signal_types': ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'],
                'signal_strengths': ['STRONG', 'MODERATE', 'WEAK'],
                'trend_types': ['golden_cross', 'death_cross'],
                'default_periods': ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                'enhanced_signal_logic': {
                    'trend_analysis': {
                        'golden_cross': '+2 points when EMA 50 > EMA 200',
                        'death_cross': '-2 points when EMA 50 < EMA 200',
                        'supertrend': '+1 if price above Supertrend, -1 if below'
                    },
                    'momentum_analysis': {
                        'rsi_ideal': '+2 points for RSI 40-60 (ideal entry zone)',
                        'rsi_oversold': '+1 for RSI 30-40 (moderate oversold)',
                        'macd_crossover': '+1 when MACD > Signal line, -1 otherwise'
                    },
                    'strength_analysis': {
                        'strong_trend': '+1 when ADX > 25',
                        'volume_confirmation': '+1 when volume > 120% of average'
                    },
                    'signal_determination': {
                        'strong_buy': 'Total score >= 4 AND positive trend',
                        'buy': 'Total score >= 2 AND neutral/positive trend',
                        'hold': 'All other conditions',
                        'sell': 'Total score <= -2',
                        'strong_sell': 'Total score <= -4 OR trend <= -2'
                    }
                },
                'api_version': '2.0_supabase',
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            return jsonify(config_info)
            
        except Exception as e:
            logger.error(f"Error in config API: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # ============= ERROR HANDLERS =============

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error: {e}")
        return render_template('500.html'), 500

    return app

# if __name__ == '__main__':
#     app = create_app()
#     logger.info("Starting Flask application with Supabase PostgreSQL backend")
#     app.run(debug=True, host='0.0.0.0', port=5097)
