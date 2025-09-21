# config.py - Supabase PostgreSQL Configuration Settings
import os
from datetime import timedelta

class Config:
    # === SUPABASE DATABASE SETTINGS ===
    # Use Supabase connection string format
    DATABASE_URL = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DATABASE_URL')
    
    # Alternative Supabase connection components (if not using full URL)
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_ANON_KEY')
    SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
    
    # Database connection pool settings for Supabase
    DB_POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', '20'))
    DB_MAX_OVERFLOW = int(os.environ.get('DB_MAX_OVERFLOW', '0'))
    DB_POOL_TIMEOUT = int(os.environ.get('DB_POOL_TIMEOUT', '30'))
    
    # === DATA REFRESH SETTINGS ===
    DATA_REFRESH_INTERVAL = timedelta(hours=1)  # Refresh data every hour
    FORCE_REFRESH_THRESHOLD = timedelta(minutes=15)  # Allow force refresh every 15 minutes
    
    # === ETF DATA SETTINGS ===
    DEFAULT_ETF_LIST = [
        'SPY', 'QQQ', 'VTI', 'IWM', 'EFA', 'EEM', 'VEA', 'IEFA', 'AGG', 'BND',
        'VNQ', 'VTEB', 'VWO', 'SCHX', 'VXUS', 'ITOT', 'IXUS', 'IEMG', 'XLK', 'XLF',
        'XLV', 'XLI', 'XLY', 'XLP', 'XLE', 'XLB', 'XLRE', 'XLU', 'GLD', 'SLV'
    ]
    
    # === TECHNICAL INDICATOR SETTINGS ===
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Enhanced moving averages
    SMA_SHORT_PERIOD = 20
    SMA_LONG_PERIOD = 50
    EMA_SHORT_PERIOD = 12
    EMA_MEDIUM_PERIOD = 26
    EMA_LONG_PERIOD = 50
    EMA_VERY_LONG_PERIOD = 200  # For Golden/Death Cross
    
    # MACD settings
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    
    # Other indicators
    ADX_PERIOD = 14
    STOCHASTIC_K_PERIOD = 14
    STOCHASTIC_D_PERIOD = 3
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    ATR_PERIOD = 14
    
    # === CHART SETTINGS ===
    CHART_THEME = 'quartz'
    DEFAULT_PERIOD = '6mo'
    AVAILABLE_PERIODS = ['1mo', '3mo', '6mo', '1y', '2y', '5y']
    
    # === API SETTINGS ===
    MAX_ETFS_OVERVIEW = 20  # Increased for Supabase performance
    MAX_ETFS_FILTER = 200   # Increased for better filtering
    REQUEST_TIMEOUT = 30    # Timeout for external API calls
    
    # Rate limiting (to prevent API abuse)
    RATE_LIMIT_DELAY = 0.1  # Seconds between Yahoo Finance requests
    MAX_RETRIES = 3         # For failed API calls
    RETRY_DELAY = 1.0       # Seconds to wait between retries
    
    # === FILE PATHS ===
    ETF_LIST_CSV = 'static/ETF_list_ETFdb_2025.csv'
    ETF_INFO_PICKLE = '100ETFS_info.pkl'
    
    # === FLASK SETTINGS ===
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # === LOGGING ===
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # === DATA QUALITY SETTINGS ===
    MIN_DATA_POINTS = 10        # Minimum data points required for analysis
    MAX_DATA_AGE_DAYS = 7       # Consider data stale after this many days
    MIN_VOLUME_THRESHOLD = 1000 # Minimum volume for reliable analysis
    
    # === SIGNAL GENERATION SETTINGS ===
    SIGNAL_CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for strong signals
    TREND_STRENGTH_THRESHOLD = 25       # ADX threshold for strong trend
    VOLUME_CONFIRMATION_RATIO = 1.2     # Volume must be 120% of average for confirmation
    
    # === SUPABASE SPECIFIC SETTINGS ===
    # Connection retry settings for Supabase
    CONNECTION_RETRY_ATTEMPTS = 3
    CONNECTION_RETRY_DELAY = 2.0
    
    # Query timeout settings
    QUERY_TIMEOUT = 60  # seconds
    BULK_INSERT_BATCH_SIZE = 1000
    
    # === ENVIRONMENT VALIDATION ===
    @classmethod
    def validate_supabase_config(cls):
        """Validate that required Supabase configuration is present"""
        required_vars = []
        
        if not cls.DATABASE_URL:
            required_vars.append('DATABASE_URL or SUPABASE_DATABASE_URL')
        
        if required_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")
        
        return True
    
    # === UTILITY METHODS ===
    @classmethod
    def get_etf_table_name(cls, ticker):
        """Generate table name for ETF data (PostgreSQL compatible)"""
        # Ensure ticker is PostgreSQL table name compliant
        clean_ticker = ticker.lower().replace('-', '_').replace('.', '_')
        return f"etf_{clean_ticker}_data"
    
    @classmethod
    def get_etf_info_table_name(cls):
        """Get table name for ETF metadata"""
        return "etf_info"
    
    @classmethod
    def get_data_refresh_log_table_name(cls):
        """Get table name for data refresh logs"""
        return "data_refresh_log"
    
    @classmethod
    def get_database_connection_string(cls):
        """Get properly formatted database connection string for Supabase"""
        if cls.DATABASE_URL:
            return cls.DATABASE_URL
        elif cls.SUPABASE_URL:
            # If individual components are provided, construct the URL
            # Note: This would need additional parsing logic
            raise ValueError("Please provide complete DATABASE_URL for Supabase connection")
        else:
            raise ValueError("No Supabase database configuration found")
    
    # === SUPABASE TABLE SCHEMAS ===
    ETF_DATA_SCHEMA = {
        'required_columns': [
            'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'
        ],
        'technical_indicators': [
            'rsi', 'rsi_vol', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 
            'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_histogram',
            'supertrend', 'parabolic_sar', 'adx', 'bb_upper', 'bb_middle', 
            'bb_lower', 'atr', 'stoch_k', 'stoch_d', 'williams_r'
        ],
        'metadata_columns': [
            'signal', 'created_at', 'updated_at'
        ]
    }

# === DEVELOPMENT/PRODUCTION ENVIRONMENT DETECTION ===
class DevelopmentConfig(Config):
    DEBUG = True
    MAX_ETFS_OVERVIEW = 20  # Smaller for development
    MAX_ETFS_FILTER = 50

class ProductionConfig(Config):
    DEBUG = False
    MAX_ETFS_OVERVIEW = 20
    MAX_ETFS_FILTER = 500
    
    # Production specific settings
    DB_POOL_SIZE = 40
    REQUEST_TIMEOUT = 60

# === CONFIGURATION FACTORY ===
def get_config():
    """Factory function to get appropriate config based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()

# Default config instance
config = get_config()

# Validate configuration on import
if __name__ != '__main__':
    try:
        Config.validate_supabase_config()
    except ValueError as e:
        print(f"Configuration Warning: {e}")
        print("Please ensure your .env file contains the required Supabase settings.")