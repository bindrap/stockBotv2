#!/usr/bin/env python3
"""
AI Trading Bot Setup and Launch Script
This script helps you set up and run the Enhanced AI Trading Bot
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print("💡 Try running: pip install -r requirements.txt manually")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        print("💡 Please create requirements.txt in the project directory")
        return False
    
    return True

def create_project_structure():
    """Create necessary directories and files"""
    print("📁 Creating project structure...")
    
    # Create directories
    directories = ['models', 'templates', 'static', 'logs', 'data']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    # Create .env file for configuration
    env_path = Path('.env')
    if not env_path.exists():
        env_content = """# AI Trading Bot Configuration
INITIAL_CAPITAL=1000
MAX_POSITION_SIZE=0.1
STOP_LOSS=0.02
TAKE_PROFIT=0.03
MIN_CONFIDENCE=0.6
MAX_DAILY_TRADES=3
DAILY_LOSS_LIMIT=0.05

# LLM Integration (Optional)
OLLAMA_URL=http://localhost:11434
CLAUDE_API_KEY=

# Interactive Brokers (for live trading)
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# Symbols to Trade (comma-separated)
SYMBOLS=AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,NFLX

# Paper Trading (True for simulation, False for live)
PAPER_TRADING=True
"""
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ Created .env configuration file")

def setup_ollama():
    """Setup instructions for Ollama"""
    print("\n🦙 Ollama Setup Instructions (Optional but Recommended):")
    print("   1. Install Ollama from: https://ollama.ai/download")
    print("   2. Run: ollama pull llama2")
    print("   3. Start server: ollama serve")
    print("   ⚠️  Ollama provides enhanced AI market analysis")

def setup_ibkr():
    """Setup instructions for Interactive Brokers"""
    print("\n📈 Interactive Brokers Setup Instructions:")
    print("   1. Download TWS or IB Gateway from IBKR")
    print("   2. Enable API in TWS: Configure → API → Enable ActiveX and Socket Clients")
    print("   3. Set Socket Port: 7497 (paper) or 7496 (live)")
    print("   4. Create Paper Trading Account for testing")
    print("   ⚠️  Start with paper trading to test the system")

def run_initial_training():
    """Run initial model training"""
    print("\n🧠 Starting initial model training...")
    print("   This may take 10-15 minutes...")
    
    try:
        # Import and run the bot training
        from bot import EnhancedAITradingBot
        
        async def train_models():
            bot = EnhancedAITradingBot(initial_capital=1000, paper_trading=True)
            
            print("   📊 Training traditional ML models...")
            bot.train_all_models()
            
            print("   💾 Saving trained models...")
            bot.save_enhanced_models('models/enhanced_trading_models.pkl')
            
            print("   📈 Running initial backtest...")
            results = bot.backtest(start_date='2023-01-01', end_date='2023-12-31')
            
            return bot
        
        bot = asyncio.run(train_models())
        print("✅ Model training completed successfully!")
        return bot
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Check that all dependencies are installed correctly")
        return None

def test_yahoo_finance():
    """Test Yahoo Finance data connection"""
    print("\n📡 Testing Yahoo Finance data connection...")
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1d")
        if not data.empty:
            print("✅ Yahoo Finance connection working")
            latest_price = data['Close'].iloc[-1]
            print(f"   AAPL latest price: ${latest_price:.2f}")
            return True
        else:
            print("❌ No data received from Yahoo Finance")
            return False
    except Exception as e:
        print(f"❌ Yahoo Finance connection failed: {e}")
        return False

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("\n📝 Creating startup scripts...")
    
    # Windows batch file
    bat_content = """@echo off
echo Starting AI Trading Bot...
python bot.py
pause
"""
    with open('start_bot.bat', 'w') as f:
        f.write(bat_content)
    
    # Unix shell script
    sh_content = """#!/bin/bash
echo "Starting AI Trading Bot..."
python bot.py
"""
    with open('start_bot.sh', 'w') as f:
        f.write(sh_content)
    
    # Make shell script executable
    try:
        os.chmod('start_bot.sh', 0o755)
    except:
        pass
    
    # Web dashboard startup
    web_content = """#!/usr/bin/env python3
import subprocess
import webbrowser
import time

print("🌐 Starting AI Trading Bot Web Dashboard...")
print("📱 Dashboard will open at: http://localhost:5000")

# Start Flask app
subprocess.Popen(["python", "app.py"])

# Wait a moment then open browser
time.sleep(3)
webbrowser.open("http://localhost:5000")
"""
    with open('start_dashboard.py', 'w') as f:
        f.write(web_content)
    
    print("✅ Created startup scripts:")
    print("   - start_bot.bat (Windows)")
    print("   - start_bot.sh (Unix/Mac)")
    print("   - start_dashboard.py (Web interface)")

def main():
    """Main setup function"""
    print("🚀 AI Trading Bot Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create project structure
    create_project_structure()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed. Please install requirements manually.")
        return
    
    # Test data connection
    test_yahoo_finance()
    
    # Create startup scripts
    create_startup_scripts()
    
    # Setup instructions
    setup_ollama()
    setup_ibkr()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed!")
    
    # Ask user what to do next
    while True:
        print("\nWhat would you like to do?")
        print("1. Train models and run backtest")
        print("2. Start web dashboard")
        print("3. Run bot in terminal mode") 
        print("4. Test system components")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            bot = run_initial_training()
            if bot:
                print("🎉 Training completed successfully!")
                print("💡 Models saved in 'models/' directory")
                print("💡 You can now start the web dashboard (option 2)")
        
        elif choice == '2':
            print("🌐 Starting web dashboard...")
            print("📱 Open your browser to: http://localhost:5000")
            try:
                subprocess.run(["python", "app.py"])
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped")
            except Exception as e:
                print(f"❌ Failed to start web dashboard: {e}")
        
        elif choice == '3':
            print("🤖 Running bot in terminal mode...")
            try:
                subprocess.run(["python", "bot.py"])
            except KeyboardInterrupt:
                print("\n👋 Bot stopped")
            except Exception as e:
                print(f"❌ Failed to run bot: {e}")
        
        elif choice == '4':
            print("🔧 Testing system components...")
            test_yahoo_finance()
            print("💡 Check logs for any issues")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()