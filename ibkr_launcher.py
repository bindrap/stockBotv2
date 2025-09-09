#!/usr/bin/env python3
"""
IBKR Trading Bot Launcher
Easy setup and launch script for the AI trading bot with Interactive Brokers
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required = ['ib_insync', 'pandas', 'numpy', 'yfinance', 'scikit-learn', 'ta', 'tensorflow']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def check_files():
    """Check if all required files exist"""
    required_files = ['ibkr_bot.py', 'requirements.txt']
    missing = []
    
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    
    print("✅ All required files found")
    return True

async def test_ibkr_connection():
    """Test IBKR connection"""
    try:
        from test_ibkr_connection import IBKRTester
        tester = IBKRTester()
        return await tester.test_connection()
    except ImportError:
        print("⚠️ IBKR test script not found, skipping connection test")
        return True

def main_menu():
    """Display main menu"""
    print("\n🤖 IBKR AI Trading Bot Launcher")
    print("=" * 40)
    print("1. 🔧 Test IBKR Connection")
    print("2. 📚 Train Models Only") 
    print("3. 🚀 Run Trading Bot")
    print("4. 📊 Web Dashboard")
    print("5. ⚙️ Setup & Configuration")
    print("6. ❌ Exit")
    print("=" * 40)

async def main():
    """Main launcher function"""
    print("🚀 IBKR AI Trading Bot")
    print("Initializing...")
    
    # Check system requirements
    if not check_requirements():
        return
    
    if not check_files():
        return
    
    while True:
        main_menu()
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\n🔌 Testing IBKR Connection...")
            try:
                result = subprocess.run([sys.executable, "test_ibkr_connection.py"], 
                                      capture_output=False, text=True)
                if result.returncode == 0:
                    print("✅ Connection test completed")
                else:
                    print("❌ Connection test failed")
            except FileNotFoundError:
                print("❌ test_ibkr_connection.py not found")
        
        elif choice == '2':
            print("\n📚 Training Models...")
            print("This will take 10-15 minutes...")
            try:
                from ibkr_bot import IBKRTradingBot
                bot = IBKRTradingBot()
                bot.train_all_models()
                print("✅ Model training completed!")
            except Exception as e:
                print(f"❌ Training failed: {e}")
        
        elif choice == '3':
            print("\n🚀 Starting Trading Bot...")
            print("Make sure TWS/IB Gateway is running!")
            
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                try:
                    subprocess.run([sys.executable, "ibkr_bot.py"])
                except KeyboardInterrupt:
                    print("\n👋 Bot stopped by user")
                except Exception as e:
                    print(f"❌ Bot failed: {e}")
        
        elif choice == '4':
            print("\n📊 Starting Web Dashboard...")
            print("Dashboard will open at: http://localhost:5000")
            
            try:
                # Check if Flask app exists
                if Path("app.py").exists():
                    subprocess.run([sys.executable, "app.py"])
                else:
                    print("❌ app.py not found")
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped")
            except Exception as e:
                print(f"❌ Dashboard failed: {e}")
        
        elif choice == '5':
            print("\n⚙️ Setup & Configuration")
            print("=" * 30)
            print("📖 IBKR Setup Steps:")
            print("1. Create IBKR account with paper trading")
            print("2. Download TWS or IB Gateway") 
            print("3. Enable API: Configure → API → Settings")
            print("4. Set port 7497 for paper trading")
            print("5. Enable 'ActiveX and Socket Clients'")
            print("6. Add 127.0.0.1 to trusted IPs")
            print("\n📁 Required Files:")
            print("- ibkr_bot.py (main bot)")
            print("- test_ibkr_connection.py (connection test)")
            print("- requirements.txt (dependencies)")
            print("- app.py (web dashboard)")
            print("\n🔗 Helpful Links:")
            print("- IBKR Account: https://www.interactivebrokers.com/")
            print("- TWS Download: https://www.interactivebrokers.com/en/trading/tws.php")
            print("- API Docs: https://interactivebrokers.github.io/tws-api/")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Launcher stopped")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")