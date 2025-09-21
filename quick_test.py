# quick_test.py - Run this to test everything quickly
import os
import sys

def quick_test():
    print("=== ETF Dashboard Quick Test ===\n")
    
    # Test 1: Check dependencies
    print("1. Testing dependencies...")
    try:
        import flask, pandas, numpy, duckdb, yfinance, altair
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return
    
    # Test 2: Test data fetching
    print("\n2. Testing data fetching...")
    try:
        from services.data_service import DataService
        service = DataService()
        data = service.fetch_etf_data('SPY', period='5d')
        if not data.empty:
            print(f"✅ Data fetched: {len(data)} records for SPY")
        else:
            print("❌ No data returned")
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
    
    # Test 3: Test database
    print("\n3. Testing database...")
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager('data/test_quick.db')
        if db.check_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    # Test 4: Test Flask app
    print("\n4. Testing Flask app...")
    try:
        from app import create_app
        app = create_app()
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Flask app working")
            else:
                print(f"❌ Flask app error: {response.status_code}")
    except Exception as e:
        print(f"❌ Flask test failed: {e}")
    
    print("\n=== Test Complete ===")
    print("If all tests pass, run: python app.py")

if __name__ == "__main__":
    quick_test()