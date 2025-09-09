# ibkr_bot.py - Enhanced bot with Interactive Brokers integration

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle
from datetime import datetime, timedelta
import logging
import warnings
import requests
import json
import asyncio
import aiohttp
import schedule
import time
import threading
import os
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
warnings.filterwarnings('ignore')

class IBKRTradingBot:
    def __init__(self, initial_capital=1000, max_position_size=0.1, stop_loss=0.02, take_profit=0.03, 
                 ib_host='127.0.0.1', ib_port=7497, client_id=1):
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # IBKR connection settings
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False
        
        # Symbols to trade
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']
        
        # AI Model Configuration
        self.models = {
            'traditional': {},
            'deep_learning': {},
            'llm_analysis': {}
        }
        self.scalers = {}
        self.feature_columns = []
        
        # Trading parameters
        self.min_confidence = 0.6
        self.max_daily_trades = 3
        self.daily_loss_limit = 0.05
        
        # Trading state
        self.positions = {}
        self.daily_trades = 0
        self.daily_pnl = 0
        self.trade_history = []
        self.account_info = {}
        
        # Training status
        self.training_status = {
            'traditional_models_trained': False,
            'deep_learning_models_trained': False,
            'training_accuracy': {},
            'last_training_time': None
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    async def connect_to_ib(self):
        """Connect to Interactive Brokers"""
        try:
            self.logger.info(f"Connecting to IBKR at {self.ib_host}:{self.ib_port}")
            await self.ib.connectAsync(self.ib_host, self.ib_port, clientId=self.client_id)
            
            # Get account info
            accounts = self.ib.managedAccounts()
            if accounts:
                self.account = accounts[0]
                self.logger.info(f"Connected to IBKR Account: {self.account}")
                
                # Check if paper trading
                if self.account.startswith('DU'):
                    self.logger.info("✅ Paper Trading Mode Active")
                else:
                    self.logger.warning("⚠️ LIVE TRADING MODE - BE CAREFUL!")
                
                self.connected = True
                await self.update_account_info()
                return True
            else:
                self.logger.error("No managed accounts found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    async def update_account_info(self):
        """Update account information"""
        if not self.connected:
            return
        
        try:
            # Get account summary
            summary = self.ib.accountSummary()
            self.account_info = {}
            
            for item in summary:
                self.account_info[item.tag] = float(item.value) if item.value.replace('.', '').replace('-', '').isdigit() else item.value
            
            # Key metrics
            self.current_capital = self.account_info.get('NetLiquidation', self.initial_capital)
            cash = self.account_info.get('TotalCashValue', 0)
            buying_power = self.account_info.get('BuyingPower', 0)
            
            self.logger.info(f"Account Value: ${self.current_capital:.2f}, Cash: ${cash:.2f}, Buying Power: ${buying_power:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating account info: {e}")
    
    async def get_current_price(self, symbol):
        """Get current market price from IBKR"""
        if not self.connected:
            return None
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            await util.sleep(1)  # Wait for data
            
            # Get price (try different price types)
            if ticker.last == ticker.last:  # Not NaN
                price = ticker.last
            elif ticker.close == ticker.close:
                price = ticker.close
            elif ticker.marketPrice == ticker.marketPrice:
                price = ticker.marketPrice()
            else:
                self.logger.warning(f"No valid price for {symbol}")
                return None
            
            # Cancel market data to avoid fees
            self.ib.cancelMktData(contract)
            
            return price
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def execute_trade(self, symbol, action, quantity, order_type='MKT'):
        """Execute trade through IBKR"""
        if not self.connected:
            self.logger.error("Not connected to IBKR")
            return False
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Create order
            if order_type == 'MKT':
                order = MarketOrder(action, quantity)
            else:  # Limit order
                current_price = await self.get_current_price(symbol)
                if not current_price:
                    return False
                
                # Set limit price (slightly favorable)
                limit_price = current_price * (0.999 if action == 'BUY' else 1.001)
                order = LimitOrder(action, quantity, limit_price)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for fill
            max_wait = 30  # seconds
            wait_time = 0
            
            while not trade.isDone() and wait_time < max_wait:
                await util.sleep(1)
                wait_time += 1
            
            if trade.isDone():
                # Get fill details
                fill_price = 0
                fill_quantity = 0
                
                for fill in trade.fills:
                    fill_price += fill.execution.price * fill.execution.shares
                    fill_quantity += fill.execution.shares
                
                if fill_quantity > 0:
                    avg_price = fill_price / fill_quantity
                    
                    # Record trade
                    trade_record = {
                        'symbol': symbol,
                        'action': action,
                        'quantity': fill_quantity,
                        'price': avg_price,
                        'timestamp': datetime.now(),
                        'order_id': trade.order.orderId
                    }
                    
                    self.trade_history.append(trade_record)
                    self.daily_trades += 1
                    
                    self.logger.info(f"✅ {action} {fill_quantity} {symbol} @ ${avg_price:.2f}")
                    
                    # Update positions
                    await self.update_positions()
                    
                    return True
                else:
                    self.logger.warning(f"Order placed but no fills received for {symbol}")
                    return False
            else:
                self.logger.warning(f"Order not filled within {max_wait}s for {symbol}")
                # Cancel unfilled order
                self.ib.cancelOrder(order)
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    async def update_positions(self):
        """Update current positions from IBKR"""
        if not self.connected:
            return
        
        try:
            positions = self.ib.positions()
            self.positions = {}
            
            for pos in positions:
                if pos.contract.secType == 'STK':  # Stocks only
                    symbol = pos.contract.symbol
                    
                    # Get current market price
                    current_price = await self.get_current_price(symbol)
                    
                    if current_price:
                        unrealized_pnl = pos.position * (current_price - pos.avgCost)
                        
                        self.positions[symbol] = {
                            'quantity': pos.position,
                            'avg_cost': pos.avgCost,
                            'current_price': current_price,
                            'market_value': pos.position * current_price,
                            'unrealized_pnl': unrealized_pnl
                        }
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def calculate_position_size(self, symbol, confidence):
        """Calculate position size based on available capital and confidence"""
        try:
            # Get available cash
            available_cash = self.account_info.get('TotalCashValue', self.current_capital * 0.5)
            
            # Calculate base position size
            base_position_value = min(
                available_cash * self.max_position_size,  # % of available cash
                self.current_capital * self.max_position_size,  # % of total portfolio
                available_cash * 0.9  # Don't use all available cash
            )
            
            # Adjust by confidence
            adjusted_position_value = base_position_value * confidence
            
            return max(100, adjusted_position_value)  # Minimum $100 position
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 100
    
    def fetch_historical_data(self, symbol, period='2y'):
        """Fetch historical data (using yfinance for training, IBKR for live prices)"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def create_features(self, data):
        """Create technical indicators and features"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_change'] = df['Close'] - df['Open']
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volume_price_ratio'] = df['Volume'] / df['Close']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}']
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['Close']
        df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Volume indicators
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Target variable
        df['target'] = df['returns'].shift(-1)
        
        return df
    
    def train_model(self, symbol):
        """Train ML model for symbol"""
        self.logger.info(f"Training model for {symbol}")
        
        # Get historical data
        data = self.fetch_historical_data(symbol, period='5y')
        if data is None:
            return False
        
        # Create features
        df = self.create_features(data)
        df = df.dropna()
        
        if len(df) < 100:
            self.logger.warning(f"Insufficient data for {symbol}")
            return False
        
        # Prepare training data
        exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        }
        
        trained_models = {}
        accuracy_scores = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            accuracy_scores[name] = {
                'train_r2': train_score,
                'test_r2': test_score,
                'mse': mean_squared_error(y_test, model.predict(X_test_scaled))
            }
            
            trained_models[name] = model
            self.logger.info(f"{symbol} {name}: Test R²={test_score:.4f}")
        
        # Store models
        self.models['traditional'][symbol] = trained_models
        self.scalers[symbol] = scaler
        self.feature_columns = feature_cols
        self.training_status['training_accuracy'][symbol] = accuracy_scores
        
        return True
    
    async def get_prediction(self, symbol):
        """Get ML prediction for symbol"""
        if symbol not in self.models['traditional']:
            return None, 0
        
        # Get recent data for features
        data = self.fetch_historical_data(symbol, period='1y')
        if data is None:
            return None, 0
        
        # Get current price from IBKR
        current_price = await self.get_current_price(symbol)
        if current_price:
            # Add current price to recent data
            latest_row = data.iloc[-1:].copy()
            latest_row.loc[latest_row.index[-1], 'Close'] = current_price
            data = pd.concat([data, latest_row])
        
        # Create features
        df = self.create_features(data)
        df = df.dropna()
        
        if len(df) == 0:
            return None, 0
        
        # Get latest features
        latest_features = df[self.feature_columns].iloc[-1:].values
        latest_features_scaled = self.scalers[symbol].transform(latest_features)
        
        # Ensemble prediction
        predictions = []
        confidences = []
        
        for model_name, model in self.models['traditional'][symbol].items():
            pred = model.predict(latest_features_scaled)[0]
            predictions.append(pred)
            
            # Confidence based on recent performance
            accuracy = self.training_status['training_accuracy'][symbol][model_name]
            confidence = max(0.3, min(0.9, accuracy['test_r2']))
            confidences.append(confidence)
        
        # Weighted average
        weights = np.array(confidences)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        final_prediction = np.average(predictions, weights=weights)
        avg_confidence = np.mean(confidences)
        
        return final_prediction, avg_confidence
    
    async def trading_routine(self):
        """Main trading routine"""
        self.logger.info("Starting trading routine")
        
        # Update account and positions
        await self.update_account_info()
        await self.update_positions()
        
        # Reset daily counters if new day
        current_date = datetime.now().date()
        if not hasattr(self, 'last_trading_date') or self.last_trading_date != current_date:
            self.daily_trades = 0
            self.daily_pnl = 0
            self.last_trading_date = current_date
        
        # Check daily limits
        if self.daily_trades >= self.max_daily_trades:
            self.logger.info("Daily trade limit reached")
            return
        
        # Analyze each symbol
        for symbol in self.symbols:
            # Skip if already have position
            if symbol in self.positions and abs(self.positions[symbol]['quantity']) > 0:
                continue
            
            try:
                # Get prediction
                prediction, confidence = await self.get_prediction(symbol)
                
                if prediction is None:
                    continue
                
                # Trading decision
                min_prediction = 0.01  # Minimum 1% expected return
                
                if abs(prediction) >= min_prediction and confidence >= self.min_confidence:
                    
                    # Calculate position size
                    position_value = self.calculate_position_size(symbol, confidence)
                    current_price = await self.get_current_price(symbol)
                    
                    if current_price and position_value >= 100:
                        quantity = int(position_value / current_price)
                        
                        if quantity > 0:
                            action = 'BUY' if prediction > 0 else 'SELL'
                            
                            self.logger.info(f"{symbol}: Prediction={prediction:.4f}, Confidence={confidence:.2f}, Action={action}")
                            
                            # Execute trade
                            success = await self.execute_trade(symbol, action, quantity)
                            
                            if success:
                                self.logger.info(f"✅ Trade executed: {action} {quantity} {symbol}")
                            else:
                                self.logger.warning(f"❌ Trade failed: {action} {quantity} {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
        
        self.logger.info("Trading routine completed")
    
    def train_all_models(self):
        """Train models for all symbols"""
        self.logger.info("Training models for all symbols...")
        
        success_count = 0
        for symbol in self.symbols:
            if self.train_model(symbol):
                success_count += 1
        
        self.training_status['traditional_models_trained'] = success_count == len(self.symbols)
        self.training_status['last_training_time'] = datetime.now()
        
        self.logger.info(f"Training completed: {success_count}/{len(self.symbols)} models")
    
    def get_status(self):
        """Get current bot status"""
        return {
            'connected': self.connected,
            'account': self.account if self.connected else 'Not Connected',
            'current_capital': self.current_capital,
            'cash': self.account_info.get('TotalCashValue', 0),
            'positions': self.positions,
            'daily_trades': self.daily_trades,
            'training_status': self.training_status,
            'models_trained': len(self.models['traditional'])
        }
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IBKR")

# Example usage
async def main():
    # Initialize bot
    bot = IBKRTradingBot(initial_capital=1000)
    
    # Connect to IBKR
    if await bot.connect_to_ib():
        print("✅ Connected to Interactive Brokers")
        
        # Train models (do this once)
        print("📚 Training models...")
        bot.train_all_models()
        
        # Get status
        status = bot.get_status()
        print(f"💰 Account: {status['account']}")
        print(f"💵 Capital: ${status['current_capital']:.2f}")
        print(f"🤖 Models: {status['models_trained']}")
        
        # Test prediction
        print("\n🔮 Testing predictions...")
        for symbol in bot.symbols[:3]:  # Test first 3
            pred, conf = await bot.get_prediction(symbol)
            if pred is not None:
                print(f"{symbol}: {pred:.4f} (confidence: {conf:.2f})")
        
        # Run trading routine
        print("\n🚀 Running trading routine...")
        await bot.trading_routine()
        
        # Show final status
        await bot.update_positions()
        status = bot.get_status()
        print(f"\n📊 Final Status:")
        print(f"Capital: ${status['current_capital']:.2f}")
        print(f"Positions: {len(status['positions'])}")
        print(f"Daily Trades: {status['daily_trades']}")
        
        # Disconnect
        bot.disconnect()
    
    else:
        print("❌ Failed to connect to Interactive Brokers")
        print("💡 Make sure TWS/Gateway is running and API is enabled")

if __name__ == "__main__":
    asyncio.run(main())