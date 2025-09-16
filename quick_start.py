#!/usr/bin/env python3
"""
Quick Start Script for IBKR Trading Bot
This script helps you get the bot running quickly with proper setup checks
"""

import asyncio
import sys
import os
from datetime import datetime

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🤖 AI STOCK TRADING BOT - INTERACTIVE BROKERS")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_requirements():
    """Check if required packages are installed"""
    print("📦 Checking requirements...")
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'ib_insync', 
        'sklearn', 'tensorflow', 'ta'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("🔧 Run: source venv/bin/activate && pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied")
    return True

async def test_connection():
    """Test IBKR connection before starting bot"""
    print("\n🔌 Testing IBKR connection...")
    
    try:
        from test_ibkr_connection import IBKRTester
        tester = IBKRTester()
        
        # Test connection only
        if await tester.test_connection():
            print("✅ IBKR connection successful")
            tester.ib.disconnect()
            return True
        else:
            print("❌ IBKR connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

async def run_bot():
    """Run the main trading bot"""
    print("\n🚀 Starting AI Trading Bot...")
    
    try:
        from ibkr_bot import IBKRTradingBot
        
        # Initialize bot
        bot = IBKRTradingBot(initial_capital=1000)
        
        # Connect to IBKR
        if not await bot.connect_to_ib():
            print("❌ Failed to connect to IBKR")
            return False
        
        print("✅ Connected to Interactive Brokers")
        
        # Check if models need training
        if not bot.training_status['traditional_models_trained']:
            print("\n📚 Training AI models (this may take a few minutes)...")
            bot.train_all_models()
            print("✅ Model training completed")
        else:
            print("✅ Using existing trained models")
        
        # Show status
        status = bot.get_status()
        print(f"\n📊 Bot Status:")
        print(f"   Account: {status['account']}")
        print(f"   Capital: ${status['current_capital']:.2f}")
        print(f"   Cash: ${status['cash']:.2f}")
        print(f"   Models: {status['models_trained']} trained")
        
        # Run a few iterations
        print(f"\n🔄 Starting trading routine...")
        for i in range(3):
            print(f"\n--- Trading Cycle {i+1} ---")
            await bot.trading_routine()
            
            # Show positions if any
            await bot.update_positions()
            if bot.positions:
                print("📈 Current Positions:")
                for symbol, pos in bot.positions.items():
                    pnl = pos['unrealized_pnl']
                    pnl_str = f"${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                    print(f"   {symbol}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f} (P&L: {pnl_str})")
            else:
                print("📊 No positions currently held")
            
            # Wait before next cycle
            if i < 2:  # Don't wait after last cycle
                print("⏳ Waiting 30 seconds before next cycle...")
                await asyncio.sleep(30)
        
        print(f"\n✅ Trading session completed")
        print(f"📊 Daily trades: {bot.daily_trades}")
        
        # Disconnect
        bot.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Bot error: {e}")
        return False

def main():
    """Main execution function"""
    print_banner()
    
    # Check if we should skip connection test
    skip_test = '--skip-test' in sys.argv
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Test connection first (unless skipped)
    if not skip_test:
        print("🧪 Running connection test first...")
        if not asyncio.run(test_connection()):
            print("\n💡 Connection failed. Make sure:")
            print("   1. TWS or IB Gateway is running")
            print("   2. API is enabled in TWS settings")
            print("   3. Port 7497 is set for paper trading")
            print("   4. You're logged into your paper trading account")
            print("\n🔧 To skip connection test, use: python quick_start.py --skip-test")
            sys.exit(1)
    
    # Run the bot
    print("\n" + "=" * 60)
    print("🎯 STARTING LIVE TRADING SESSION")
    print("=" * 60)
    
    try:
        success = asyncio.run(run_bot())
        
        if success:
            print("\n🎉 Trading session completed successfully!")
            print("💡 Check your TWS for positions and trades")
        else:
            print("\n❌ Trading session failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Trading session stopped by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()