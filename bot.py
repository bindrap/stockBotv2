import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
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
warnings.filterwarnings('ignore')

# Paper trading simulation (since IB connection might not be available initially)
class PaperTradingSimulator:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        
    def get_current_price(self, symbol):
        """Get current price using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return data['Close'].iloc[-1]
            else:
                # Fallback to daily data
                data = ticker.history(period='5d')
                return data['Close'].iloc[-1]
        except:
            return None
    
    def place_order(self, symbol, action, quantity, price=None):
        """Simulate placing an order"""
        if price is None:
            price = self.get_current_price(symbol)
        
        if price is None:
            return False
        
        order_value = quantity * price
        
        if action.upper() == 'BUY':
            if order_value <= self.current_capital:
                self.current_capital -= order_value
                if symbol in self.positions:
                    # Average down
                    old_qty = self.positions[symbol]['quantity']
                    old_price = self.positions[symbol]['avg_price']
                    new_qty = old_qty + quantity
                    new_avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty
                    self.positions[symbol] = {
                        'quantity': new_qty,
                        'avg_price': new_avg_price,
                        'market_value': new_qty * price
                    }
                else:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'market_value': quantity * price
                    }
                return True
            else:
                return False
        
        elif action.upper() == 'SELL':
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                self.current_capital += quantity * price
                self.positions[symbol]['quantity'] -= quantity
                
                # Calculate P&L
                pnl = quantity * (price - self.positions[symbol]['avg_price'])
                
                # Record trade
                trade_record = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'timestamp': datetime.now(),
                    'pnl': pnl
                }
                self.trade_history.append(trade_record)
                
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
                else:
                    self.positions[symbol]['market_value'] = self.positions[symbol]['quantity'] * price
                
                return True
            else:
                return False
        
        return False
    
    def get_portfolio_value(self):
        """Calculate total portfolio value"""
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                total_value += position['quantity'] * current_price
        
        return total_value
    
    def get_positions_summary(self):
        """Get current positions with current market values"""
        summary = {}
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                unrealized_pnl = position['quantity'] * (current_price - position['avg_price'])
                summary[symbol] = {
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'current_price': current_price,
                    'market_value': position['quantity'] * current_price,
                    'unrealized_pnl': unrealized_pnl
                }
        return summary

class EnhancedAITradingBot:
    def __init__(self, initial_capital=1000, max_position_size=0.1, stop_loss=0.02, take_profit=0.03, paper_trading=True):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.paper_trading = paper_trading
        
        # Initialize paper trading simulator
        if self.paper_trading:
            self.broker = PaperTradingSimulator(initial_capital)
        
        # Expanded watchlist
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']
        
        # AI Model Configuration
        self.models = {
            'traditional': {},  # RF, GB models
            'deep_learning': {},  # LSTM models
            'llm_analysis': {}  # LLM-based market analysis
        }
        self.scalers = {}
        self.feature_columns = []
        
        # LLM Integration Setup
        self.ollama_url = "http://localhost:11434"
        self.claude_api_key = None
        self.llm_enabled = False
        
        # Trading parameters
        self.min_confidence = 0.6
        self.max_daily_trades = 3
        self.daily_loss_limit = 0.05
        
        # Continuous Learning Parameters
        self.learning_schedule = {
            'retrain_frequency': 'weekly',
            'data_lookback': 252,
            'performance_threshold': 0.6,
            'adaptation_rate': 0.1
        }
        
        # Market Hours Learning
        self.after_hours_learning = True
        self.learning_queue = []
        self.performance_metrics = []
        
        # Trading state
        self.positions = {}
        self.daily_trades = 0
        self.daily_pnl = 0
        self.trade_history = []
        
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

    def fetch_historical_data(self, symbol, period='2y'):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def create_features(self, data):
        """Create comprehensive technical indicators and features"""
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
        
        # Volatility indicators
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Technical indicators using ta library
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
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Volume indicators
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
        
        # Market regime indicators
        df['trend_strength'] = abs(df['Close'].rolling(window=20).apply(lambda x: np.polyfit(range(20), x, 1)[0]))
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Target variable (next day return)
        df['target'] = df['returns'].shift(-1)
        
        return df

    def prepare_training_data(self, symbol):
        """Prepare training data with proper train/validation split"""
        data = self.fetch_historical_data(symbol, period='5y')
        if data is None:
            return None, None, None, None
        
        # Create features
        df = self.create_features(data)
        df = df.dropna()
        
        if len(df) < 100:
            self.logger.warning(f"Insufficient data for {symbol}: {len(df)} samples")
            return None, None, None, None
        
        # Select feature columns (exclude target and non-predictive columns)
        exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Time series split (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.feature_columns = feature_cols
        
        return X_train, X_test, y_train, y_test

    def train_model(self, symbol):
        """Train ensemble model for a specific symbol"""
        self.logger.info(f"Training traditional models for {symbol}")
        
        X_train, X_test, y_train, y_test = self.prepare_training_data(symbol)
        if X_train is None:
            return False
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble of models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        }
        
        trained_models = {}
        accuracy_scores = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            accuracy_scores[name] = {
                'train_r2': train_score,
                'test_r2': test_score,
                'mse': mse,
                'mae': mae
            }
            
            self.logger.info(f"{symbol} - {name}: Train R²={train_score:.4f}, Test R²={test_score:.4f}, MSE={mse:.6f}")
            
            trained_models[name] = model
        
        # Store models and scaler
        self.models['traditional'][symbol] = trained_models
        self.scalers[symbol] = scaler
        self.training_status['training_accuracy'][symbol] = accuracy_scores
        
        return True

    def get_prediction(self, symbol):
        """Get prediction for a symbol using traditional ML ensemble"""
        if symbol not in self.models['traditional']:
            return None, 0
        
        # Get latest data
        data = self.fetch_historical_data(symbol, period='1y')
        if data is None:
            return None, 0
        
        # Create features
        df = self.create_features(data)
        df = df.dropna()
        
        if len(df) == 0:
            return None, 0
        
        # Get latest features
        latest_features = df[self.feature_columns].iloc[-1:].values
        
        # Scale features
        latest_features_scaled = self.scalers[symbol].transform(latest_features)
        
        # Ensemble prediction
        predictions = []
        confidences = []
        
        for model_name, model in self.models['traditional'][symbol].items():
            pred = model.predict(latest_features_scaled)[0]
            predictions.append(pred)
            
            # Simple confidence based on feature importance (placeholder)
            feature_importance = getattr(model, 'feature_importances_', None)
            if feature_importance is not None:
                confidence = min(0.9, max(0.3, np.mean(feature_importance) * 5))
            else:
                confidence = 0.6
            confidences.append(confidence)
        
        # Weighted ensemble
        weights = np.array(confidences)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        final_prediction = np.average(predictions, weights=weights)
        avg_confidence = np.mean(confidences)
        
        return final_prediction, avg_confidence

    def setup_llm_integration(self, claude_api_key=None, ollama_url=None):
        """Setup LLM integration for market analysis"""
        if claude_api_key:
            self.claude_api_key = claude_api_key
        
        if ollama_url:
            self.ollama_url = ollama_url
        
        # Test connections
        self.test_llm_connections()

    def test_llm_connections(self):
        """Test LLM connections"""
        # Test Ollama connection
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("Ollama connection successful")
                self.llm_enabled = True
            else:
                self.logger.warning("Ollama not available")
        except Exception as e:
            self.logger.warning(f"Ollama connection failed: {e}")
        
        # Test Claude API
        if self.claude_api_key:
            try:
                self.logger.info("Claude API key configured")
                self.llm_enabled = True
            except Exception as e:
                self.logger.warning(f"Claude API connection failed: {e}")

    async def llm_market_analysis(self, symbol, market_data, news_data=None):
        """Use LLM for market analysis and sentiment"""
        if not self.llm_enabled:
            return None
        
        # Prepare context for LLM
        context = f"""
        Analyze the following market data for {symbol}:
        
        Current Price: ${market_data['current_price']:.2f}
        Price Change: {market_data['price_change']:.2%}
        Volume: {market_data['volume']:,}
        RSI: {market_data['rsi']:.2f}
        MACD: {market_data['macd']:.4f}
        
        Technical Indicators Summary:
        - Moving Average (20): ${market_data['sma_20']:.2f}
        - Bollinger Bands Position: {market_data['bb_position']:.2f}
        - Volume Ratio: {market_data['volume_ratio']:.2f}
        
        Recent Performance:
        - 1-day return: {market_data['return_1d']:.2%}
        - 5-day return: {market_data['return_5d']:.2%}
        - 20-day volatility: {market_data['volatility_20d']:.2%}
        """
        
        if news_data:
            context += f"\n\nRecent News Headlines:\n{news_data}"
        
        prompt = f"""
        {context}
        
        Based on this data, provide:
        1. Market sentiment (bullish/bearish/neutral) with confidence score
        2. Key technical levels to watch
        3. Risk factors to consider
        4. Trading recommendation (buy/sell/hold) with reasoning
        
        Respond in JSON format with sentiment_score (-1 to 1), confidence (0 to 1), 
        recommendation, and reasoning.
        """
        
        try:
            if self.ollama_url:
                analysis = await self.query_ollama(prompt)
                return self.parse_llm_response(analysis)
        except Exception as e:
            self.logger.error(f"LLM analysis failed for {symbol}: {e}")
            return None

    async def query_ollama(self, prompt, model="llama2"):
        """Query Ollama for market analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
                
                async with session.post(f"{self.ollama_url}/api/generate", 
                                      json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        self.logger.error(f"Ollama API error: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {e}")
            return None

    def parse_llm_response(self, response):
        """Parse LLM response into structured data"""
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                response_lower = response.lower()
                if 'bullish' in response_lower or 'buy' in response_lower:
                    sentiment = 0.5
                elif 'bearish' in response_lower or 'sell' in response_lower:
                    sentiment = -0.5
                else:
                    sentiment = 0.0
                
                return {
                    'sentiment_score': sentiment,
                    'confidence': 0.6,
                    'recommendation': 'hold',
                    'reasoning': response[:200] + '...' if len(response) > 200 else response
                }
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return None

    def build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM training"""
        df = self.create_features(data)
        df = df.dropna()
        
        # Select features for LSTM
        feature_cols = [col for col in df.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X, y = [], []
        for i in range(lookback, len(df)):
            X.append(df[feature_cols].iloc[i-lookback:i].values)
            y.append(df['target'].iloc[i])
        
        return np.array(X), np.array(y), feature_cols

    def train_deep_learning_models(self, symbol):
        """Train LSTM and other deep learning models"""
        self.logger.info(f"Training deep learning models for {symbol}")
        
        data = self.fetch_historical_data(symbol, period='5y')
        if data is None:
            return False
        
        X, y, feature_cols = self.prepare_lstm_data(data)
        
        if len(X) < 100:
            self.logger.warning(f"Insufficient data for LSTM training: {symbol}")
            return False
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=0.0001),
            ModelCheckpoint(f'models/lstm_{symbol}_best.h5', save_best_only=True)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        
        self.logger.info(f"{symbol} LSTM - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        self.models['deep_learning'][symbol] = {
            'model': model,
            'feature_cols': feature_cols,
            'lookback': 60,
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        
        return True

    async def get_enhanced_prediction(self, symbol):
        """Get enhanced prediction using all AI models"""
        predictions = {}
        confidences = {}
        
        # Traditional ML prediction
        traditional_pred, traditional_conf = self.get_prediction(symbol)
        if traditional_pred is not None:
            predictions['traditional'] = traditional_pred
            confidences['traditional'] = traditional_conf
        
        # Deep learning prediction
        if symbol in self.models['deep_learning']:
            dl_pred, dl_conf = await self.get_lstm_prediction(symbol)
            if dl_pred is not None:
                predictions['deep_learning'] = dl_pred
                confidences['deep_learning'] = dl_conf
        
        # LLM analysis
        if self.llm_enabled:
            market_data = self.get_current_market_data(symbol)
            if market_data:
                llm_analysis = await self.llm_market_analysis(symbol, market_data)
                if llm_analysis:
                    predictions['llm'] = llm_analysis['sentiment_score'] * 0.05
                    confidences['llm'] = llm_analysis['confidence']
        
        # Ensemble prediction
        if predictions:
            total_weight = sum(confidences.values())
            if total_weight > 0:
                weighted_pred = sum(pred * confidences[model] for model, pred in predictions.items()) / total_weight
                avg_confidence = sum(confidences.values()) / len(confidences)
                
                return weighted_pred, avg_confidence, predictions, confidences
        
        return None, 0, {}, {}

    async def get_lstm_prediction(self, symbol):
        """Get prediction from LSTM model"""
        if symbol not in self.models['deep_learning']:
            return None, 0
        
        try:
            model_data = self.models['deep_learning'][symbol]
            model = model_data['model']
            feature_cols = model_data['feature_cols']
            lookback = model_data['lookback']
            
            data = self.fetch_historical_data(symbol, period='1y')
            df = self.create_features(data)
            df = df.dropna()
            
            if len(df) < lookback:
                return None, 0
            
            recent_data = df[feature_cols].tail(lookback).values
            X = recent_data.reshape(1, lookback, len(feature_cols))
            
            prediction = model.predict(X, verbose=0)[0][0]
            
            # Confidence based on model performance
            test_loss = model_data.get('test_loss', 0.01)
            confidence = max(0.3, min(0.9, 1 / (1 + test_loss * 100)))
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"LSTM prediction error for {symbol}: {e}")
            return None, 0

    def get_current_market_data(self, symbol):
        """Get current market data for LLM analysis"""
        try:
            data = self.fetch_historical_data(symbol, period='3mo')
            if data is None or len(data) == 0:
                return None
            
            df = self.create_features(data)
            df = df.dropna()
            
            if len(df) == 0:
                return None
            
            latest = df.iloc[-1]
            
            return {
                'current_price': latest['Close'],
                'price_change': latest['returns'],
                'volume': latest['Volume'],
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'sma_20': latest['sma_20'],
                'bb_position': latest['bb_position'],
                'volume_ratio': latest['volume_ratio'],
                'return_1d': latest['returns'],
                'return_5d': df['returns'].tail(5).sum(),
                'volatility_20d': df['returns'].tail(20).std()
            }
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    def setup_continuous_learning(self):
        """Setup continuous learning schedule"""
        schedule.every().day.at("16:30").do(self.after_hours_learning_routine)
        schedule.every().sunday.at("10:00").do(self.weekly_model_update)
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        self.logger.info("Continuous learning scheduler started")

    def after_hours_learning_routine(self):
        """Run learning routine after market hours"""
        self.logger.info("Starting after-hours learning routine")
        
        self.analyze_daily_performance()
        self.incremental_model_update()

    def analyze_daily_performance(self):
        """Analyze daily trading performance and learn"""
        if not self.trade_history:
            return
        
        today_trades = [trade for trade in self.trade_history 
                       if trade['timestamp'].date() == datetime.now().date()]
        
        if not today_trades:
            return
        
        total_pnl = sum(trade['pnl'] for trade in today_trades)
        win_rate = len([t for t in today_trades if t['pnl'] > 0]) / len(today_trades)
        
        performance_data = {
            'date': datetime.now().date(),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'num_trades': len(today_trades),
            'trades': today_trades
        }
        
        self.performance_metrics.append(performance_data)
        
        if win_rate < 0.5 or total_pnl < 0:
            self.learning_queue.extend([trade['symbol'] for trade in today_trades])
        
        self.logger.info(f"Daily performance: P&L=${total_pnl:.2f}, Win Rate={win_rate:.2%}")

    def incremental_model_update(self):
        """Incrementally update models with new data"""
        if not self.learning_queue:
            return
        
        symbols_to_update = list(set(self.learning_queue))
        
        for symbol in symbols_to_update:
            try:
                self.logger.info(f"Incrementally updating models for {symbol}")
                self.train_model(symbol)
                
            except Exception as e:
                self.logger.error(f"Error updating models for {symbol}: {e}")
        
        self.learning_queue.clear()

    def weekly_model_update(self):
        """Weekly comprehensive model update"""
        self.logger.info("Starting weekly model update")
        
        for symbol in self.symbols:
            self.train_model(symbol)
            self.train_deep_learning_models(symbol)
        
        self.save_enhanced_models('models/enhanced_trading_models.pkl')

    def risk_management_check(self, symbol, signal):
        """Comprehensive risk management"""
        portfolio_value = self.broker.get_portfolio_value() if self.paper_trading else self.current_capital
        
        if self.daily_pnl <= -self.daily_loss_limit * portfolio_value:
            self.logger.warning("Daily loss limit reached. No more trades today.")
            return False
        
        if self.daily_trades >= self.max_daily_trades:
            self.logger.warning("Maximum daily trades reached.")
            return False
        
        return True

    async def execute_trade(self, symbol, signal, confidence):
        """Execute trade through paper trading or real broker"""
        if not self.risk_management_check(symbol, signal):
            return False
        
        try:
            current_price = self.broker.get_current_price(symbol)
            if current_price is None:
                self.logger.error(f"No price data for {symbol}")
                return False
            
            portfolio_value = self.broker.get_portfolio_value()
            position_value = portfolio_value * self.max_position_size * confidence
            quantity = int(position_value / current_price)
            
            if quantity == 0:
                self.logger.warning(f"Calculated quantity is 0 for {symbol}")
                return False
            
            action = 'BUY' if signal > 0 else 'SELL'
            
            if action == 'SELL' and symbol not in self.broker.positions:
                self.logger.warning(f"Cannot sell {symbol} - no position held")
                return False
            
            success = self.broker.place_order(symbol, action, quantity)
            
            if success:
                self.positions[symbol] = {
                    'quantity': quantity if action == 'BUY' else -quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'stop_loss': current_price * (1 - self.stop_loss) if action == 'BUY' else current_price * (1 + self.stop_loss),
                    'take_profit': current_price * (1 + self.take_profit) if action == 'BUY' else current_price * (1 - self.take_profit)
                }
                
                self.daily_trades += 1
                self.logger.info(f"Executed {action} order for {quantity} shares of {symbol} at ${current_price:.2f}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return False

    def monitor_positions(self):
        """Monitor existing positions for stop loss/take profit"""
        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.broker.get_current_price(symbol)
                if current_price is None:
                    continue
                
                should_close = False
                reason = ""
                
                if position['quantity'] > 0:  # Long position
                    if current_price <= position['stop_loss']:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price >= position['take_profit']:
                        should_close = True
                        reason = "Take Profit"
                else:  # Short position
                    if current_price >= position['stop_loss']:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price <= position['take_profit']:
                        should_close = True
                        reason = "Take Profit"
                
                if should_close:
                    self.close_position(symbol, reason)
                    
            except Exception as e:
                self.logger.error(f"Error monitoring position for {symbol}: {e}")

    def close_position(self, symbol, reason="Manual Close"):
        """Close a specific position"""
        if symbol not in self.positions:
            return False
        
        try:
            position = self.positions[symbol]
            action = 'SELL' if position['quantity'] > 0 else 'BUY'
            quantity = abs(position['quantity'])
            
            success = self.broker.place_order(symbol, action, quantity)
            
            if success:
                del self.positions[symbol]
                self.logger.info(f"Closed position for {symbol}. Reason: {reason}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return False

    async def daily_routine(self):
        """Main daily trading routine"""
        self.logger.info("Starting daily trading routine")
        
        self.daily_trades = 0
        self.daily_pnl = 0
        
        self.monitor_positions()
        
        for symbol in self.symbols:
            if symbol in self.positions:
                continue
            
            try:
                prediction, confidence, all_preds, all_confs = await self.get_enhanced_prediction(symbol)
                
                if prediction is None:
                    continue
                
                if abs(prediction) > 0.01 and confidence > self.min_confidence:
                    signal = 1 if prediction > 0 else -1
                    
                    self.logger.info(f"{symbol}: Prediction={prediction:.4f}, Confidence={confidence:.4f}, Signal={signal}")
                    
                    await self.execute_trade(symbol, signal, confidence)
            
            except Exception as e:
                self.logger.error(f"Error in daily routine for {symbol}: {e}")
        
        portfolio_value = self.broker.get_portfolio_value()
        self.logger.info(f"Daily routine completed. Trades: {self.daily_trades}, Portfolio Value: ${portfolio_value:.2f}")

    def train_all_models(self):
        """Train all models for all symbols"""
        self.logger.info("Starting comprehensive model training...")
        
        # Train traditional models
        traditional_success = 0
        for symbol in self.symbols:
            if self.train_model(symbol):
                traditional_success += 1
        
        self.training_status['traditional_models_trained'] = traditional_success == len(self.symbols)
        
        # Train deep learning models
        dl_success = 0
        for symbol in self.symbols:
            if self.train_deep_learning_models(symbol):
                dl_success += 1
        
        self.training_status['deep_learning_models_trained'] = dl_success > 0
        self.training_status['last_training_time'] = datetime.now()
        
        self.logger.info(f"Training completed. Traditional: {traditional_success}/{len(self.symbols)}, Deep Learning: {dl_success}/{len(self.symbols)}")

    def get_status(self):
        """Get current bot status"""
        portfolio_value = self.broker.get_portfolio_value() if self.paper_trading else self.current_capital
        positions_summary = self.broker.get_positions_summary() if self.paper_trading else {}
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.broker.current_capital if self.paper_trading else self.current_capital,
            'positions': positions_summary,
            'daily_trades': self.daily_trades,
            'training_status': self.training_status,
            'performance_metrics': self.performance_metrics[-7:] if self.performance_metrics else [],
            'models_trained': {
                'traditional': len(self.models['traditional']),
                'deep_learning': len(self.models['deep_learning']),
                'llm_enabled': self.llm_enabled
            }
        }

    def get_training_accuracy(self):
        """Get training accuracy for all models"""
        return self.training_status['training_accuracy']

    def save_enhanced_models(self, filepath):
        """Save all models including deep learning"""
        model_data = {
            'traditional_models': self.models['traditional'],
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics,
            'learning_parameters': self.learning_schedule,
            'training_status': self.training_status
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        for symbol, model_data in self.models['deep_learning'].items():
            model_data['model'].save(f'models/lstm_model_{symbol}.h5')
        
        self.logger.info(f"Models saved to {filepath}")

    def load_enhanced_models(self, filepath):
        """Load all saved models"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models['traditional'] = model_data['traditional_models']
            self.scalers = model_data['scalers']
            self.feature_columns = model_data['feature_columns']
            self.performance_metrics = model_data.get('performance_metrics', [])
            self.learning_schedule = model_data.get('learning_parameters', self.learning_schedule)
            self.training_status = model_data.get('training_status', self.training_status)
            
            for symbol in self.symbols:
                try:
                    model_path = f'models/lstm_model_{symbol}.h5'
                    if os.path.exists(model_path):
                        model = load_model(model_path)
                        self.models['deep_learning'][symbol] = {'model': model}
                except:
                    pass
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def backtest(self, start_date='2020-01-01', end_date='2023-12-31'):
        """Run comprehensive backtest"""
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        
        backtest_results = []
        
        for symbol in self.symbols:
            data = self.fetch_historical_data(symbol, period='5y')
            if data is None:
                continue
            
            df = self.create_features(data)
            df = df.dropna()
            
            df = df.loc[start_date:end_date]
            
            if len(df) < 50:
                continue
            
            returns = []
            for i in range(50, len(df)):
                current_data = df.iloc[i]
                
                signal = 0
                if current_data['rsi'] < 30 and current_data['bb_position'] < 0.2:
                    signal = 1
                elif current_data['rsi'] > 70 and current_data['bb_position'] > 0.8:
                    signal = -1
                
                actual_return = current_data['target']
                strategy_return = signal * actual_return
                returns.append(strategy_return)
            
            if returns:
                total_return = np.prod([1 + r for r in returns]) - 1
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                max_drawdown = self.calculate_max_drawdown(returns)
                
                backtest_results.append({
                    'symbol': symbol,
                    'total_return': total_return,
                    'annualized_return': (1 + total_return) ** (252 / len(returns)) - 1,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'num_trades': len([r for r in returns if r != 0])
                })
        
        self.display_backtest_results(backtest_results)
        return backtest_results

    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def display_backtest_results(self, results):
        """Display backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\nSymbol: {result['symbol']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Annualized Return: {result['annualized_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Number of Trades: {result['num_trades']}")
        
        if results:
            avg_return = np.mean([r['annualized_return'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            
            print(f"\nOVERALL STATISTICS")
            print(f"Average Annualized Return: {avg_return:.2%}")
            print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print("="*60)


# Usage example and main execution
async def main():
    # Initialize enhanced bot
    print("🤖 Initializing Enhanced AI Trading Bot...")
    bot = EnhancedAITradingBot(initial_capital=1000, paper_trading=True)
    
    # Setup LLM integration (optional)
    try:
        bot.setup_llm_integration(ollama_url="http://localhost:11434")
        print("✅ LLM integration setup completed")
    except:
        print("⚠️ LLM integration not available (continuing without it)")
    
    # Setup continuous learning
    bot.setup_continuous_learning()
    print("✅ Continuous learning scheduler started")
    
    # Train all models
    print("\n📚 Starting model training...")
    bot.train_all_models()
    
    # Run backtest
    print("\n📈 Running backtest...")
    backtest_results = bot.backtest()
    
    # Test enhanced prediction
    print("\n🔮 Testing enhanced predictions...")
    for symbol in bot.symbols[:3]:
        try:
            pred, conf, all_preds, all_confs = await bot.get_enhanced_prediction(symbol)
            if pred is not None:
                print(f"{symbol}: Prediction={pred:.4f}, Confidence={conf:.2f}")
                if all_preds:
                    print(f"  Model breakdown: {all_preds}")
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
    
    # Save models
    bot.save_enhanced_models('models/enhanced_trading_models.pkl')
    print("✅ Models saved successfully")
    
    # Show current status
    print("\n📊 Current Bot Status:")
    status = bot.get_status()
    print(f"Portfolio Value: ${status['portfolio_value']:.2f}")
    print(f"Cash: ${status['cash']:.2f}")
    print(f"Models Trained: {status['models_trained']}")
    
    # Optional: Start daily routine
    response = input("\n🚀 Start daily trading routine? (y/n): ")
    if response.lower() == 'y':
        await bot.daily_routine()
    
    return bot

if __name__ == "__main__":
    import asyncio
    bot = asyncio.run(main())