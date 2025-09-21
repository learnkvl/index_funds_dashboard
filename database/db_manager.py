# database/db_manager.py - Supabase PostgreSQL Database Manager

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from contextlib import contextmanager
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    @staticmethod
    def get_etf_table_name(ticker: str) -> str:
        # Replace invalid characters for PostgreSQL table names
        return f"etf_{ticker.lower().replace('-', '_')}_data"

class DatabaseManager:
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("Database connection string not provided. Set DATABASE_URL environment variable.")
        
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database and create tables"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create ETF info table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS etf_info (
                            ticker VARCHAR PRIMARY KEY,
                            name VARCHAR,
                            aum BIGINT,
                            description TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create data refresh log table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS data_refresh_log (
                            ticker VARCHAR,
                            refresh_type VARCHAR,
                            start_date DATE,
                            end_date DATE,
                            records_count INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (ticker, refresh_type, start_date, end_date)
                        )
                    """)
                    
                    conn.commit()
                    logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def check_connection(self) -> bool:
        """Check if database connection is working"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def create_etf_table(self, ticker: str):
        """Create table for specific ETF data with enhanced technical indicators"""
        table_name = Config.get_etf_table_name(ticker)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            date DATE PRIMARY KEY,
                            open DOUBLE PRECISION,
                            high DOUBLE PRECISION,
                            low DOUBLE PRECISION,
                            close DOUBLE PRECISION,
                            volume BIGINT,
                            adj_close DOUBLE PRECISION,
                            
                            -- Basic RSI indicators
                            rsi DOUBLE PRECISION,
                            rsi_vol DOUBLE PRECISION,
                            
                            -- Enhanced Moving averages for signal generation
                            sma_20 DOUBLE PRECISION,
                            sma_50 DOUBLE PRECISION,
                            ema_12 DOUBLE PRECISION,
                            ema_26 DOUBLE PRECISION,
                            ema_50 DOUBLE PRECISION,           -- Critical for Golden Cross
                            ema_200 DOUBLE PRECISION,          -- Critical for Death Cross
                            
                            -- MACD system (essential for momentum signals)
                            macd DOUBLE PRECISION,
                            macd_signal DOUBLE PRECISION,
                            macd_histogram DOUBLE PRECISION,
                            
                            -- Trend indicators
                            supertrend DOUBLE PRECISION,
                            parabolic_sar DOUBLE PRECISION,
                            
                            -- Strength and momentum indicators
                            adx DOUBLE PRECISION,              -- Trend strength
                            
                            -- Ichimoku components
                            tenkan_sen DOUBLE PRECISION,
                            kijun_sen DOUBLE PRECISION,
                            senkou_span_a DOUBLE PRECISION,
                            senkou_span_b DOUBLE PRECISION,
                            chikou_span DOUBLE PRECISION,
                            
                            -- Additional indicators
                            linear_regression DOUBLE PRECISION,
                            
                            -- Bollinger Bands
                            bb_upper DOUBLE PRECISION,
                            bb_middle DOUBLE PRECISION,
                            bb_lower DOUBLE PRECISION,
                            
                            -- Volatility
                            atr DOUBLE PRECISION,
                            
                            -- Stochastic
                            stoch_k DOUBLE PRECISION,
                            stoch_d DOUBLE PRECISION,
                            
                            -- Williams %R
                            williams_r DOUBLE PRECISION,
                            
                            -- Rate of change and momentum
                            roc DOUBLE PRECISION,
                            momentum DOUBLE PRECISION,
                            
                            -- Volume indicators (critical for signal confirmation)
                            volume_sma DOUBLE PRECISION,
                            
                            -- Enhanced trading signal
                            signal VARCHAR,
                            
                            -- Metadata
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
                    logger.info(f"Created/updated table for {ticker} with enhanced technical indicators")
                
        except Exception as e:
            logger.error(f"Error creating table for {ticker}: {e}")
            raise

    def insert_etf_data(self, ticker: str, df: pd.DataFrame) -> int:
        """Insert ETF data into database with enhanced technical indicators"""
        if df.empty:
            return 0

        table_name = Config.get_etf_table_name(ticker)
        self.create_etf_table(ticker)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Normalize column names
                    df_clean = df.copy()
                    df_clean.columns = df_clean.columns.str.lower()

                    # Ensure 'date' exists
                    if 'date' not in df_clean.columns:
                        df_clean.reset_index(inplace=True)
                        if 'date' not in df_clean.columns and 'Date' in df_clean.columns:
                            df_clean.rename(columns={'Date': 'date'}, inplace=True)

                    # Remove duplicate dates (keep last)
                    df_clean = df_clean.drop_duplicates(subset=['date'], keep='last')

                    # Define all expected columns that match our enhanced schema
                    expected_cols = [
                        "date", "open", "high", "low", "close", "volume", "adj_close",
                        "rsi", "rsi_vol", 
                        "sma_20", "sma_50", "ema_12", "ema_26", "ema_50", "ema_200",
                        "macd", "macd_signal", "macd_histogram", 
                        "supertrend", "parabolic_sar", "adx",
                        "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span",
                        "linear_regression", "bb_upper", "bb_middle", "bb_lower", "atr",
                        "stoch_k", "stoch_d", "williams_r", "roc", "momentum", "volume_sma", "signal"
                    ]
                    
                    # Keep only existing columns from expected columns
                    df_clean = df_clean[[col for col in expected_cols if col in df_clean.columns]]

                    # Add metadata
                    df_clean['created_at'] = datetime.now()
                    df_clean['updated_at'] = datetime.now()

                    # Convert DataFrame to list of tuples for batch insert
                    columns = list(df_clean.columns)
                    values = [tuple(row) for row in df_clean.values]

                    # Delete existing records for overlapping dates first
                    if 'date' in df_clean.columns:
                        dates = df_clean['date'].tolist()
                        date_placeholders = ','.join(['%s'] * len(dates))
                        cur.execute(f"""
                            DELETE FROM {table_name} 
                            WHERE date IN ({date_placeholders})
                        """, dates)

                    # Prepare insert statement
                    cols_str = ", ".join(columns)
                    placeholders = ", ".join(["%s"] * len(columns))
                    insert_query = f"""
                        INSERT INTO {table_name} ({cols_str})
                        VALUES ({placeholders})
                    """

                    # Execute batch insert
                    cur.executemany(insert_query, values)
                    conn.commit()

                    inserted_count = len(values)
                    logger.info(f"Inserted {inserted_count} records for {ticker} with enhanced indicators")
                    return inserted_count

        except Exception as e:
            logger.error(f"Error inserting enhanced data for {ticker}: {e}")
            return 0
            
    def get_etf_data(self, ticker: str, start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve ETF data from database"""
        table_name = Config.get_etf_table_name(ticker)
        
        try:
            with self.get_connection() as conn:
                # Build query
                query = f"SELECT * FROM {table_name}"
                params = []
                
                where_conditions = []
                if start_date:
                    where_conditions.append("date >= %s")
                    params.append(start_date.date())
                
                if end_date:
                    where_conditions.append("date <= %s")
                    params.append(end_date.date())
                
                if where_conditions:
                    query += " WHERE " + " AND ".join(where_conditions)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                return df
                
        except Exception as e:
            if "does not exist" in str(e).lower() or "relation" in str(e).lower():
                logger.info(f"Table for {ticker} doesn't exist yet")
                return pd.DataFrame()
            else:
                logger.error(f"Error retrieving data for {ticker}: {e}")
                raise
    
    def get_latest_date(self, ticker: str) -> Optional[datetime]:
        """Get the latest date for which we have data for an ETF"""
        table_name = Config.get_etf_table_name(ticker)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT MAX(date) FROM {table_name}")
                    result = cur.fetchone()
                    
                    if result and result[0]:
                        return pd.to_datetime(result[0])
                    return None
                
        except Exception as e:
            if "does not exist" in str(e).lower() or "relation" in str(e).lower():
                return None
            else:
                logger.error(f"Error getting latest date for {ticker}: {e}")
                return None
    
    def get_missing_dates(self, ticker: str, start_date: datetime, 
                         end_date: datetime) -> List[datetime]:
        """Find dates missing in the database for a given period"""
        existing_data = self.get_etf_data(ticker, start_date, end_date)
        
        if existing_data.empty:
            # Generate all business days in the range
            all_dates = pd.bdate_range(start=start_date, end=end_date)
            return all_dates.tolist()
        
        # Find gaps in existing data
        all_dates = pd.bdate_range(start=start_date, end=end_date)
        existing_dates = set(existing_data.index.date)
        missing_dates = [d for d in all_dates if d.date() not in existing_dates]
        
        return missing_dates
    
    def upsert_etf_info(self, ticker: str, name: str = None, aum: int = None, 
                       description: str = None):
        """Insert or update ETF metadata using PostgreSQL UPSERT"""
        
        def clean_aum(aum_value):
            if isinstance(aum_value, str):
                # Remove $ sign, commas, spaces, and decimal part
                cleaned = aum_value.replace('$', '').replace(',', '').replace(' ', '').strip()
                if '.' in cleaned:
                    cleaned = cleaned.split('.')[0]
                try:
                    return int(cleaned)
                except Exception as e:
                    logger.error(f"Could not clean AUM value '{aum_value}': {e}")
                    return None
            return aum_value

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Clean aum before using
                    if aum is not None:
                        aum = clean_aum(aum)

                    # Use PostgreSQL UPSERT (ON CONFLICT)
                    cur.execute("""
                        INSERT INTO etf_info (ticker, name, aum, description, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (ticker) 
                        DO UPDATE SET 
                            name = COALESCE(EXCLUDED.name, etf_info.name),
                            aum = COALESCE(EXCLUDED.aum, etf_info.aum),
                            description = COALESCE(EXCLUDED.description, etf_info.description),
                            updated_at = CURRENT_TIMESTAMP
                    """, [ticker, name, aum, description])
                    
                    conn.commit()
                    logger.info(f"Updated ETF info for {ticker}")

        except Exception as e:
            logger.error(f"Error updating ETF info for {ticker}: {e}")
            raise
    
    def get_etf_info(self, ticker: str = None) -> pd.DataFrame:
        """Get ETF metadata"""
        try:
            with self.get_connection() as conn:
                if ticker:
                    query = "SELECT * FROM etf_info WHERE ticker = %s"
                    params = [ticker]
                else:
                    query = "SELECT * FROM etf_info ORDER BY ticker"
                    params = []
                
                df = pd.read_sql_query(query, conn, params=params)
                return df
                
        except Exception as e:
            if "does not exist" in str(e).lower() or "relation" in str(e).lower():
                return pd.DataFrame()
            else:
                logger.error(f"Error getting ETF info: {e}")
                raise
    
    def log_data_refresh(self, ticker: str, refresh_type: str, start_date: datetime,
                        end_date: datetime, records_count: int):
        """Log data refresh activity"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO data_refresh_log 
                        (ticker, refresh_type, start_date, end_date, records_count, created_at)
                        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (ticker, refresh_type, start_date, end_date) 
                        DO UPDATE SET 
                            records_count = EXCLUDED.records_count,
                            created_at = CURRENT_TIMESTAMP
                    """, [ticker, refresh_type, start_date.date(), end_date.date(), records_count])
                    conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging data refresh: {e}")
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old refresh logs"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                    cur.execute("""
                        DELETE FROM data_refresh_log 
                        WHERE created_at < %s
                    """, [cutoff_date])
                    conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")

if __name__ == "__main__":
    # Test the enhanced database manager
    try:
        db = DatabaseManager()
        print("Enhanced PostgreSQL Database manager initialized successfully!")
        print(f"Connection test: {db.check_connection()}")
    except Exception as e:
        print(f"Error initializing database: {e}")
        print("Make sure to set DATABASE_URL environment variable")