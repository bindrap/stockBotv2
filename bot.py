import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
from ib_insync import IB, Stock, MarketOrder, util
warnings.filterwarnings('ignore')

class EnhancedAITradingBot:
    def __init__(self, initial_capital=1000, max_position_size=0.1, stop_loss=0.02, take_profit=0.03):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Expanded watchlist
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']
        
        # AI Model Configuration
        self.models = {
            'traditional': {},  # RF, GB models
            'deep_learning': {},  # LSTM, Transformer models
            'llm_analysis': {}  # LLM-based market analysis
        }
        self.scalers = {}
        self.feature_columns = []
        
        # LLM Integration Setup
        self.ollama_url = "http://localhost:11434"  # Default Ollama URL
        self.claude_api_key = None  # Will be set via environment variable
        self.llm_enabled = False
        
        # Continuous Learning Parameters
        self.learning_schedule = {
            'retrain_frequency': 'weekly',  # daily, weekly, monthly
            'data_lookback': 252,  # Trading days for training
            'performance_threshold': 0.6,  # Retrain if accuracy drops below this
            'adaptation_rate': 0.1  # How quickly to adapt to new patterns
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
        
        # Interactive Brokers
        self.ib = IB()
        
        # News and Sentiment Analysis
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.reuters.com/business/finance'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

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
                # Test API key validity (you'd implement actual Claude API call here)
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
            # Use Ollama for analysis
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
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback: simple sentiment analysis
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
        
        # Get extended historical data
        data = self.fetch_historical_data(symbol, period='5y')
        if data is None:
            return False
        
        # Prepare LSTM data
        X, y, feature_cols = self.prepare_lstm_data(data)
        
        if len(X) < 100:  # Need minimum samples
            self.logger.warning(f"Insufficient data for LSTM training: {symbol}")
            return False
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train LSTM
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=0.0001)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate model
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
        
        self.logger.info(f"{symbol} LSTM - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Store model
        self.models['deep_learning'][symbol] = {
            'model': model,
            'feature_cols': feature_cols,
            'lookback': 60
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
            llm_analysis = await self.llm_market_analysis(symbol, market_data)
            if llm_analysis:
                predictions['llm'] = llm_analysis['sentiment_score'] * 0.05  # Convert to expected return
                confidences['llm'] = llm_analysis['confidence']
        
        # Ensemble prediction
        if predictions:
            # Weight predictions by confidence
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
            
            # Get recent data
            data = self.fetch_historical_data(symbol, period='1y')
            df = self.create_features(data)
            df = df.dropna()
            
            if len(df) < lookback:
                return None, 0
            
            # Prepare input sequence
            recent_data = df[feature_cols].tail(lookback).values
            X = recent_data.reshape(1, lookback, len(feature_cols))
            
            # Make prediction
            prediction = model.predict(X, verbose=0)[0][0]
            
            # Estimate confidence based on recent accuracy
            # (You'd implement a more sophisticated confidence calculation)
            confidence = 0.7  # Placeholder
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"LSTM prediction error for {symbol}: {e}")
            return None, 0

    def get_current_market_data(self, symbol):
        """Get current market data for LLM analysis"""
        try:
            data = self.fetch_historical_data(symbol, period='3mo')
            df = self.create_features(data)
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
            return {}

    def setup_continuous_learning(self):
        """Setup continuous learning schedule"""
        # Schedule after-hours learning
        schedule.every().day.at("16:30").do(self.after_hours_learning_routine)  # After market close
        schedule.every().sunday.at("10:00").do(self.weekly_model_update)  # Weekly retraining
        
        # Start scheduler in separate thread
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
        
        # Analyze today's performance
        self.analyze_daily_performance()
        
        # Update models with new data
        self.incremental_model_update()
        
        # Analyze market conditions for tomorrow
        self.prepare_next_day_analysis()

    def analyze_daily_performance(self):
        """Analyze daily trading performance and learn"""
        if not self.trade_history:
            return
        
        today_trades = [trade for trade in self.trade_history 
                       if trade['entry_time'].date() == datetime.now().date()]
        
        if not today_trades:
            return
        
        # Calculate performance metrics
        total_pnl = sum(trade['pnl'] for trade in today_trades)
        win_rate = len([t for t in today_trades if t['pnl'] > 0]) / len(today_trades)
        avg_return = total_pnl / self.current_capital
        
        # Store performance data
        performance_data = {
            'date': datetime.now().date(),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'num_trades': len(today_trades),
            'trades': today_trades
        }
        
        self.performance_metrics.append(performance_data)
        
        # Add to learning queue if performance was poor
        if win_rate < 0.5 or avg_return < -0.02:
            self.learning_queue.extend([trade['symbol'] for trade in today_trades])
        
        self.logger.info(f"Daily performance: P&L=${total_pnl:.2f}, Win Rate={win_rate:.2%}, Return={avg_return:.2%}")

    def incremental_model_update(self):
        """Incrementally update models with new data"""
        if not self.learning_queue:
            return
        
        # Get unique symbols that need retraining
        symbols_to_update = list(set(self.learning_queue))
        
        for symbol in symbols_to_update:
            try:
                self.logger.info(f"Incrementally updating models for {symbol}")
                
                # Retrain traditional models
                self.train_model(symbol)
                
                # Retrain deep learning models
                self.train_deep_learning_models(symbol)
                
            except Exception as e:
                self.logger.error(f"Error updating models for {symbol}: {e}")
        
        # Clear learning queue
        self.learning_queue.clear()

    async def prepare_next_day_analysis(self):
        """Prepare analysis for next trading day"""
        self.logger.info("Preparing next day analysis")
        
        for symbol in self.symbols:
            try:
                # Get enhanced prediction for tomorrow
                pred, conf, all_preds, all_confs = await self.get_enhanced_prediction(symbol)
                
                if pred is not None:
                    self.logger.info(f"{symbol} tomorrow: Prediction={pred:.4f}, Confidence={conf:.2f}")
                    
                    # Store analysis for tomorrow's trading
                    analysis = {
                        'symbol': symbol,
                        'prediction': pred,
                        'confidence': conf,
                        'model_predictions': all_preds,
                        'model_confidences': all_confs,
                        'analysis_time': datetime.now()
                    }
                    
                    # You could store this in a database or file for tomorrow's use
                    
            except Exception as e:
                self.logger.error(f"Error in next day analysis for {symbol}: {e}")

    def weekly_model_update(self):
        """Weekly comprehensive model update"""
        self.logger.info("Starting weekly model update")
        
        # Retrain all models with latest data
        for symbol in self.symbols:
            self.train_model(symbol)
            self.train_deep_learning_models(symbol)
        
        # Analyze weekly performance
        self.analyze_weekly_performance()
        
        # Save updated models
        self.save_enhanced_models('enhanced_trading_models.pkl')

    def analyze_weekly_performance(self):
        """Analyze weekly performance and adjust parameters"""
        if len(self.performance_metrics) < 5:  # Need at least 5 days
            return
        
        recent_performance = self.performance_metrics[-7:]  # Last 7 days
        
        avg_return = np.mean([p['avg_return'] for p in recent_performance])
        avg_win_rate = np.mean([p['win_rate'] for p in recent_performance])
        volatility = np.std([p['avg_return'] for p in recent_performance])
        
        self.logger.info(f"Weekly performance: Avg Return={avg_return:.2%}, Win Rate={avg_win_rate:.2%}, Volatility={volatility:.2%}")
        
        # Adjust risk parameters based on performance
        if avg_return < -0.01:  # Poor performance
            self.max_position_size = max(0.05, self.max_position_size * 0.8)  # Reduce position size
            self.min_confidence = min(0.8, self.min_confidence + 0.1)  # Increase confidence threshold
        elif avg_return > 0.02 and volatility < 0.015:  # Good stable performance
            self.max_position_size = min(0.15, self.max_position_size * 1.1)  # Slightly increase position size
            self.min_confidence = max(0.5, self.min_confidence - 0.05)  # Slightly decrease confidence threshold

    def save_enhanced_models(self, filepath):
        """Save all models including deep learning"""
        model_data = {
            'traditional_models': self.models['traditional'],
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics,
            'learning_parameters': self.learning_schedule
        }
        
        # Save traditional models
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save deep learning models separately
        for symbol, model_data in self.models['deep_learning'].items():
            model_data['model'].save(f'lstm_model_{symbol}.h5')

    def load_enhanced_models(self, filepath):
        """Load all saved models"""
        # Load traditional models
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models['traditional'] = model_data['traditional_models']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.performance_metrics = model_data.get('performance_metrics', [])
        self.learning_schedule = model_data.get('learning_parameters', self.learning_schedule)
        
        # Load deep learning models
        for symbol in self.symbols:
            try:
                model = load_model(f'lstm_model_{symbol}.h5')
                self.models['deep_learning'][symbol] = {'model': model}
            except:
                pass  # Model doesn't exist yet

# Usage example
async def main():
    # Initialize enhanced bot
    bot = EnhancedAITradingBot(initial_capital=1000)
    
    # Setup LLM integration
    bot.setup_llm_integration(
        claude_api_key="your_claude_api_key_here",  # Optional
        ollama_url="http://localhost:11434"  # Make sure Ollama is running
    )
    
    # Setup continuous learning
    bot.setup_continuous_learning()
    
    # Train all models
    print("Training traditional models...")
    for symbol in bot.symbols:
        bot.train_model(symbol)
    
    print("Training deep learning models...")
    for symbol in bot.symbols:
        bot.train_deep_learning_models(symbol)
    
    # Test enhanced prediction
    print("\nTesting enhanced predictions...")
    for symbol in bot.symbols[:2]:  # Test first 2 symbols
        pred, conf, all_preds, all_confs = await bot.get_enhanced_prediction(symbol)
        print(f"{symbol}: Prediction={pred:.4f}, Confidence={conf:.2f}")
        print(f"  Model breakdown: {all_preds}")
    
    # Save models
    bot.save_enhanced_models('enhanced_trading_models.pkl')
    print("\nEnhanced models saved!")

if __name__ == "__main__":
    asyncio.run(main())