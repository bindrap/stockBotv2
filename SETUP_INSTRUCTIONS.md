# 🤖 AI Stock Trading Bot - Setup Instructions

## Prerequisites

1. **Interactive Brokers Account**
   - Open a paper trading account at [Interactive Brokers](https://www.interactivebrokers.com)
   - Download and install TWS (Trader Workstation) or IB Gateway

2. **Python Environment**
   - Python 3.8+ required
   - Virtual environment recommended

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install required packages
pip install -r requirements.txt
```

## Step 2: Configure Interactive Brokers

### TWS/Gateway Setup:
1. **Launch TWS or IB Gateway**
   - Use your IBKR credentials to log in
   - **IMPORTANT:** Switch to Paper Trading mode (top-right corner in TWS)

2. **Enable API Access:**
   - Go to `Configure` → `API` → `Settings`
   - Check "Enable ActiveX and Socket Clients"
   - Set Socket port to `7497` (paper trading) or `7496` (live)
   - Add `127.0.0.1` to "Trusted IPs"
   - Optionally check "Read-Only API" for safety

3. **Paper Trading Verification:**
   - Your paper account should start with "DU" (e.g., DU123456)
   - Live accounts start with "U" - BE VERY CAREFUL!

## Step 3: Test Connection

```bash
# Test IBKR connection
python test_ibkr_connection.py

# Test with custom port
python test_ibkr_connection.py 7497
```

Expected output:
```
✅ Connected to IBKR successfully!
✅ Paper Trading Mode Active
✅ Account Value: $1,000,000.00
```

## Step 4: Start the Bot

### Quick Start (Recommended):
```bash
python quick_start.py
```

### Manual Start:
```bash
python ibkr_bot.py
```

## Bot Features

🧠 **AI Models:**
- Random Forest & Gradient Boosting
- Technical indicators (RSI, MACD, Bollinger Bands)
- Real-time market data integration

📊 **Risk Management:**
- Maximum 10% position size per trade
- Stop loss: 2%, Take profit: 3%
- Maximum 3 trades per day
- Daily loss limit: 5%

🎯 **Symbols Traded:**
- AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, NFLX

## Safety Features

⚠️ **Paper Trading First:**
- Always start with paper trading
- Verify all functionality before considering live trading
- Bot automatically detects paper vs live accounts

🛡️ **Risk Controls:**
- Position size limits
- Daily trade limits
- Stop-loss protection
- Paper trading verification

## Monitoring

The bot provides real-time status updates:
- Connection status
- Account information
- Trade executions
- Position updates
- Model predictions

## Troubleshooting

### Connection Issues:
```
❌ Connection failed
```
**Solutions:**
1. Verify TWS/Gateway is running
2. Check API settings are enabled
3. Confirm correct port (7497 for paper)
4. Add 127.0.0.1 to trusted IPs

### Permission Issues:
```
❌ No trading permissions
```
**Solutions:**
1. Verify paper trading mode is active
2. Check account permissions in TWS
3. Ensure API trading is enabled

### Missing Data:
```
❌ No market data
```
**Solutions:**
1. Check market hours (9:30 AM - 4:00 PM ET)
2. Verify data subscriptions in TWS
3. Test with different symbols

## Important Notes

🚨 **NEVER use live trading without thorough testing**
🚨 **Always verify paper trading mode before starting**
🚨 **Monitor the bot actively during operation**
🚨 **Start with small amounts even in paper trading**

## Support

- Check logs for detailed error messages
- Use `python test_ibkr_connection.py` for diagnostics
- Verify TWS/Gateway status and settings

---

**Remember:** This is educational software. Use at your own risk and never trade with money you cannot afford to lose.