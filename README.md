# 🤖 Enhanced AI Trading Bot with Interactive Brokers Integration

## 📋 Project Overview

This is a sophisticated AI-powered trading bot that combines multiple machine learning models with real-time Interactive Brokers integration. The bot features continuous learning, web-based monitoring, and comprehensive risk management.

### **🧠 Three-Layer AI Architecture:**

1. **Traditional ML**: Random Forest + Gradient Boosting (Fast, Reliable Pattern Recognition)
2. **Deep Learning**: LSTM Neural Networks (Advanced Temporal Pattern Analysis)
3. **Large Language Models**: Market Sentiment & Context Analysis (Optional)

### **🔥 Key Features:**

- ✅ **Real-time IBKR Integration** - Connect to Interactive Brokers for live data and trading
- ✅ **Paper Trading Support** - Safe testing with IBKR paper trading accounts
- ✅ **Web Dashboard** - Real-time monitoring and control interface
- ✅ **Multi-Model AI** - Ensemble predictions from multiple AI models
- ✅ **Continuous Learning** - Models retrain after market hours
- ✅ **Risk Management** - Built-in stop-loss, position sizing, and daily limits
- ✅ **30+ Technical Indicators** - Comprehensive feature engineering
- ✅ **Backtesting** - Historical performance validation

---

## 🚀 Quick Start Guide

### **Prerequisites:**

1. **Python 3.8+** installed
2. **Interactive Brokers account** with paper trading enabled
3. **TWS or IB Gateway** downloaded and configured

### **Step 1: Installation**

```bash
# Clone or download the repository
cd your-trading-bot-directory

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: IBKR Setup**

1. **Download TWS/IB Gateway** from Interactive Brokers
2. **Login** to your paper trading account (account starts with "DU")
3. **Enable API**: Configure → API → Settings
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Socket port: **7497** (paper trading)
   - ✅ Add **127.0.0.1** to Trusted IPs
4. **Apply settings** and keep TWS/Gateway running

### **Step 3: Test Connection**

```bash
# Test your IBKR connection
python test_ibkr_connection.py
```

**Expected Output:**
```
✅ Connected to IBKR successfully!
📊 Connected to account: ['DU1234567']
✅ Paper Trading Mode Active
💰 Account Summary:
   NetLiquidation: $1,000,000.00
   TotalCashValue: $1,000,000.00
   BuyingPower: $4,000,000.00
✅ IBKR Setup Test Completed!
```

### **Step 4: Choose Your Interface**

**Option A: Web Dashboard (Recommended)**
```bash
python app.py
# Open browser to: http://localhost:5000
```

**Option B: IBKR Bot (Terminal)**
```bash
python ibkr_bot.py
```

**Option C: Launcher Menu**
```bash
python ibkr_launcher.py
# Interactive menu with all options
```

---

## 📁 File Structure

```
trading-bot/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Setup and installation script
│
├── 🤖 Core Bot Files
├── bot.py                       # Original bot (Yahoo Finance + Paper Trading)
├── ibkr_bot.py                  # IBKR-enhanced bot (REAL-TIME DATA)
├── app.py                       # Flask web dashboard
│
├── 🔧 Utility Scripts
├── test_ibkr_connection.py      # IBKR connection tester
├── ibkr_launcher.py             # Interactive launcher menu
│
├── 🎨 Web Interface
├── templates/
│   └── dashboard.html           # Web dashboard UI
│
└── 📊 Generated Directories
    ├── models/                  # Trained AI models
    ├── logs/                    # Application logs
    └── data/                    # Historical data cache
```

---

## 🎯 Recommended Workflow

### **For IBKR Paper Trading (Recommended):**

1. **Test Connection First:**
   ```bash
   python test_ibkr_connection.py
   ```

2. **Start Web Dashboard:**
   ```bash
   python app.py
   # Dashboard: http://localhost:5000
   ```

3. **In Web Dashboard:**
   - Click **"Train Models"** (takes 10-15 minutes)
   - Click **"Start Bot"** once training completes
   - Monitor performance in real-time

### **For Yahoo Finance Simulation:**

1. **Run Original Bot:**
   ```bash
   python bot.py
   ```

2. **Or Web Dashboard:**
   ```bash
   python app.py
   # Uses Yahoo Finance data with paper trading simulation
   ```

---

## 🎛️ Configuration Options

### **Conservative Setup (Recommended for Beginners):**
```python
# In ibkr_bot.py or bot.py
bot = IBKRTradingBot(
    initial_capital=1000,
    max_position_size=0.05,    # 5% max per trade
    stop_loss=0.015,           # 1.5% stop loss  
    take_profit=0.025,         # 2.5% take profit
    min_confidence=0.75,       # Higher confidence threshold
    max_daily_trades=2         # Max 2 trades per day
)
```

### **Balanced Setup (Default):**
```python
bot = IBKRTradingBot(
    initial_capital=1000,
    max_position_size=0.1,     # 10% max per trade
    stop_loss=0.02,            # 2% stop loss
    take_profit=0.03,          # 3% take profit
    min_confidence=0.6,        # Standard confidence
    max_daily_trades=3         # Max 3 trades per day
)
```

### **Aggressive Setup (Advanced Users):**
```python
bot = IBKRTradingBot(
    initial_capital=1000,
    max_position_size=0.15,    # 15% max per trade
    stop_loss=0.025,           # 2.5% stop loss
    take_profit=0.04,          # 4% take profit
    min_confidence=0.5,        # Lower confidence threshold
    max_daily_trades=5         # Max 5 trades per day
)
```

---

## 📊 Web Dashboard Features

Access at **http://localhost:5000** when running `python app.py`:

### **Real-Time Monitoring:**
- 📈 **Live Performance Charts** - P&L and win rate tracking
- 🔮 **AI Predictions** - Current model predictions for all stocks
- 💼 **Position Management** - View and close positions
- 📋 **Trade History** - Complete trade log with P&L
- 🧠 **Model Performance** - Training accuracy and metrics

### **Control Panel:**
- ▶️ **Start/Stop Bot** - Real-time bot control
- 🧠 **Train Models** - Retrain AI models with latest data
- 📈 **Run Backtests** - Historical performance testing
- ⚙️ **Real-time Status** - Connection and account monitoring

---

## 🤖 AI Model Details

### **1. Traditional Machine Learning**
- **Random Forest**: Pattern recognition and feature importance
- **Gradient Boosting**: Sequential learning and error correction
- **Training Data**: 5 years of historical stock data
- **Features**: 30+ technical indicators (RSI, MACD, Bollinger Bands, etc.)

### **2. Deep Learning (LSTM)**
- **Architecture**: 3-layer LSTM with dropout and batch normalization
- **Lookback**: 60-day time series sequences
- **Purpose**: Capture complex temporal patterns
- **Training**: Automatic early stopping and learning rate reduction

### **3. Large Language Models (Optional)**
- **Local LLM**: Ollama integration for market sentiment
- **Cloud LLM**: Claude API for advanced market analysis
- **Features**: News sentiment, market context, qualitative factors

### **Ensemble Approach:**
The bot combines all three model types using confidence-weighted averaging for final trading decisions.

---

## ⚡ Performance & Expectations

### **Realistic Performance Targets:**
- **Annual Return**: 15-30% (excellent performance)
- **Monthly Return**: 1-3% (sustainable growth)
- **Daily Return**: 0.1-0.5% average (realistic expectation)
- **Win Rate**: 55-65% (professional level)
- **Sharpe Ratio**: 1.0-2.0 (risk-adjusted returns)
- **Max Drawdown**: <15% (risk management)

### **Your $1000 Goal Timeline:**
- **3 months**: $1000 → $1150 (15% return) ✅ **Realistic**
- **6 months**: $1000 → $1300 (30% return) ✅ **Good Performance**  
- **12 months**: $1000 → $2000 (100% return) ⚠️ **Very Ambitious**

**Note**: The bot's 5% daily return goal is unrealistic. Professional traders achieve 15-30% annually.

---

## 🛡️ Risk Management Features

### **Built-in Safety Systems:**
- **Position Sizing**: Maximum 10% of portfolio per trade
- **Stop Loss**: Automatic 2% stop loss on all positions
- **Take Profit**: Automatic 3% profit taking
- **Daily Limits**: Maximum 3 trades per day
- **Loss Protection**: Stop trading if daily loss > 5%
- **Account Verification**: Confirms paper trading mode

### **Continuous Monitoring:**
- **Real-time Position Tracking**: Monitor all open positions
- **Performance Analysis**: Daily performance evaluation
- **Model Retraining**: Weekly model updates with new data
- **Risk Alerts**: Automatic warnings for unusual behavior

---

## 🔄 Continuous Learning System

### **After-Hours Learning (4:30 PM - 9:30 AM):**
1. **Performance Analysis**: Analyze today's trades and outcomes
2. **Model Updates**: Retrain underperforming models
3. **Strategy Optimization**: Adjust parameters based on results
4. **Next Day Preparation**: Generate predictions for tomorrow

### **Weekly Deep Learning (Sundays):**
1. **Comprehensive Retraining**: Update all models with latest data
2. **Performance Review**: Analyze weekly trading metrics
3. **Parameter Adjustment**: Optimize risk management settings
4. **Model Validation**: Ensure model accuracy and reliability

---

## 🚨 Important Warnings & Best Practices

### **Before Live Trading:**
- ⚠️ **Paper trade for 30+ days minimum**
- ⚠️ **Start with small amounts ($100-200, not $1000)**
- ⚠️ **Monitor closely for first week**
- ⚠️ **Understand all risk management features**
- ⚠️ **Keep TWS/Gateway running during market hours**

### **Market Data Limitations:**
- **Yahoo Finance**: 15-20 minute delay (good for testing)
- **IBKR**: Real-time or minimal delay (professional trading)
- **Market Hours**: Bot works during market hours (9:30 AM - 4:00 PM ET)

### **System Requirements:**
- **Stable Internet**: Critical for real-time trading
- **Always-On Computer**: Bot needs to run during market hours
- **Backup Power**: UPS recommended for reliability

---

## 🔧 Troubleshooting

### **Common Issues:**

**1. IBKR Connection Failed:**
```bash
# Check TWS/Gateway is running
# Verify API settings: Configure → API → Settings
# Confirm port 7497 for paper trading
# Add 127.0.0.1 to Trusted IPs
```

**2. "No managed accounts found":**
```bash
# Restart TWS/Gateway
# Check login credentials
# Verify paper trading account is active
```

**3. "No market data":**
```bash
# Normal during after-hours/weekends
# Check market data permissions
# Verify IBKR account has data subscriptions
```

**4. Web dashboard won't load:**
```bash
pip install Flask plotly
python app.py
# Open: http://localhost:5000
```

### **Getting Help:**
- **IBKR API Documentation**: https://interactivebrokers.github.io/tws-api/
- **ib_insync Documentation**: https://ib-insync.readthedocs.io/
- **Check logs**: Look in `logs/` directory for detailed error messages

---

## 🎯 Next Steps

1. **✅ Install Dependencies**: `pip install -r requirements.txt`
2. **✅ Setup IBKR**: Configure TWS/Gateway with API enabled
3. **✅ Test Connection**: `python test_ibkr_connection.py`
4. **✅ Start Dashboard**: `python app.py` → http://localhost:5000
5. **✅ Train Models**: Click "Train Models" in web dashboard (15 min)
6. **✅ Start Trading**: Click "Start Bot" and monitor performance
7. **✅ Monitor Daily**: Check positions and performance regularly

### **Recommended Learning Path:**
1. **Week 1**: Setup and paper trading with basic models
2. **Week 2**: Monitor performance and understand AI predictions  
3. **Week 3**: Experiment with different risk parameters
4. **Week 4**: Add LLM integration for enhanced analysis
5. **Month 2+**: Consider small live trading amounts if performance is good

---

## 📈 Advanced Features

### **LLM Integration Setup (Optional):**
```bash
# Install Ollama for local LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2
ollama serve

# Or use Claude API (set environment variable)
export ANTHROPIC_API_KEY="your_api_key_here"
```

### **Multiple Model Training:**
```bash
# Train all models for all symbols
python -c "
from ibkr_bot import IBKRTradingBot
bot = IBKRTradingBot()
bot.train_all_models()  # Traditional ML + LSTM for all 8 stocks
"
```

### **Custom Strategy Development:**
- Modify `create_features()` to add custom indicators
- Adjust `trading_routine()` for different trading strategies
- Update `risk_management_check()` for custom risk rules

---

## 🏆 Success Metrics

### **Key Performance Indicators:**
- **Sharpe Ratio > 1.0**: Risk-adjusted returns
- **Maximum Drawdown < 15%**: Risk management effectiveness
- **Win Rate > 55%**: Prediction accuracy
- **Consistent Daily Performance**: Steady growth pattern

### **Monthly Review Checklist:**
- [ ] Analyze monthly P&L and win rate
- [ ] Review model prediction accuracy
- [ ] Assess risk management effectiveness
- [ ] Update training data with latest market conditions
- [ ] Optimize parameters based on performance

---

**Remember**: This is a sophisticated trading system. Start conservative, understand the technology, and never risk more than you can afford to lose. The AI learns and improves over time, so patience and careful monitoring are key to success! 🚀