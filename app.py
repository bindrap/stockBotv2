from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import asyncio
import threading
import time
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from bot import EnhancedAITradingBot

app = Flask(__name__)

# Global bot instance
bot = None
bot_thread = None
bot_running = False

def initialize_bot():
    """Initialize the trading bot"""
    global bot
    bot = EnhancedAITradingBot(initial_capital=1000, paper_trading=True)
    
    # Setup LLM integration
    try:
        bot.setup_llm_integration(ollama_url="http://localhost:11434")
    except:
        print("LLM integration not available")
    
    # Setup continuous learning
    bot.setup_continuous_learning()
    
    return bot

def run_bot_loop():
    """Run the bot in a separate thread"""
    global bot_running
    bot_running = True
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while bot_running:
        try:
            # Run daily routine
            loop.run_until_complete(bot.daily_routine())
            
            # Wait for next iteration (could be every hour during market hours)
            time.sleep(3600)  # 1 hour
            
        except Exception as e:
            print(f"Error in bot loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current bot status"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    status = bot.get_status()
    status['bot_running'] = bot_running
    status['current_time'] = datetime.now().isoformat()
    
    return jsonify(status)

@app.route('/api/training_accuracy')
def get_training_accuracy():
    """Get training accuracy for all models"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    accuracy = bot.get_training_accuracy()
    return jsonify(accuracy)

@app.route('/api/performance_chart')
def get_performance_chart():
    """Get performance chart data"""
    if bot is None or not bot.performance_metrics:
        return jsonify({'error': 'No performance data available'})
    
    # Prepare data for chart
    dates = [p['date'].isoformat() if hasattr(p['date'], 'isoformat') else str(p['date']) 
             for p in bot.performance_metrics]
    pnl_values = [p['total_pnl'] for p in bot.performance_metrics]
    win_rates = [p['win_rate'] * 100 for p in bot.performance_metrics]
    
    # Calculate cumulative P&L
    cumulative_pnl = []
    running_total = 0
    for pnl in pnl_values:
        running_total += pnl
        cumulative_pnl.append(running_total)
    
    # Create Plotly chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pnl,
        mode='lines+markers',
        name='Cumulative P&L ($)',
        line=dict(color='#00D4AA', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=win_rates,
        mode='lines+markers',
        name='Win Rate (%)',
        yaxis='y2',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.update_layout(
        title='Trading Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative P&L ($)',
        yaxis2=dict(
            title='Win Rate (%)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_dark'
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'chart': graphJSON})

@app.route('/api/positions')
def get_positions():
    """Get current positions"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    if bot.paper_trading:
        positions = bot.broker.get_positions_summary()
    else:
        positions = {}
    
    return jsonify(positions)

@app.route('/api/trade_history')
def get_trade_history():
    """Get recent trade history"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    # Get last 50 trades
    recent_trades = bot.broker.trade_history[-50:] if bot.paper_trading else []
    
    # Format trades for display
    formatted_trades = []
    for trade in recent_trades:
        formatted_trades.append({
            'symbol': trade['symbol'],
            'action': trade['action'],
            'quantity': trade['quantity'],
            'price': round(trade['price'], 2),
            'timestamp': trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'pnl': round(trade['pnl'], 2)
        })
    
    return jsonify(formatted_trades)

@app.route('/api/predictions')
def get_predictions():
    """Get current predictions for all symbols"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    async def get_all_predictions():
        predictions = {}
        for symbol in bot.symbols:
            try:
                pred, conf, all_preds, all_confs = await bot.get_enhanced_prediction(symbol)
                if pred is not None:
                    predictions[symbol] = {
                        'prediction': round(pred, 4),
                        'confidence': round(conf, 2),
                        'model_breakdown': {k: round(v, 4) for k, v in all_preds.items()},
                        'confidence_breakdown': {k: round(v, 2) for k, v in all_confs.items()}
                    }
            except Exception as e:
                predictions[symbol] = {'error': str(e)}
        return predictions
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    predictions = loop.run_until_complete(get_all_predictions())
    loop.close()
    
    return jsonify(predictions)

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start model training"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    def training_thread():
        bot.train_all_models()
    
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started'})

@app.route('/api/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global bot_thread, bot_running
    
    if bot is None:
        initialize_bot()
    
    if not bot_running:
        bot_thread = threading.Thread(target=run_bot_loop)
        bot_thread.daemon = True
        bot_thread.start()
        
        return jsonify({'message': 'Bot started successfully'})
    else:
        return jsonify({'message': 'Bot is already running'})

@app.route('/api/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_running
    bot_running = False
    
    return jsonify({'message': 'Bot stopped successfully'})

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    data = request.get_json()
    start_date = data.get('start_date', '2020-01-01')
    end_date = data.get('end_date', '2023-12-31')
    
    def backtest_thread():
        bot.backtest(start_date=start_date, end_date=end_date)
    
    thread = threading.Thread(target=backtest_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'Backtest started for {start_date} to {end_date}'})

@app.route('/api/model_performance')
def get_model_performance():
    """Get detailed model performance metrics"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    performance_data = {}
    
    # Traditional model performance
    for symbol, accuracy in bot.training_status.get('training_accuracy', {}).items():
        performance_data[symbol] = {
            'traditional_models': accuracy,
            'deep_learning_available': symbol in bot.models['deep_learning'],
            'last_prediction_time': 'N/A'
        }
        
        if symbol in bot.models['deep_learning']:
            dl_model = bot.models['deep_learning'][symbol]
            performance_data[symbol]['deep_learning'] = {
                'train_loss': dl_model.get('train_loss', 'N/A'),
                'test_loss': dl_model.get('test_loss', 'N/A')
            }
    
    return jsonify(performance_data)

@app.route('/api/close_position', methods=['POST'])
def close_position():
    """Close a specific position"""
    if bot is None:
        return jsonify({'error': 'Bot not initialized'})
    
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Symbol not provided'})
    
    success = bot.close_position(symbol, "Manual Close via Web Interface")
    
    if success:
        return jsonify({'message': f'Position for {symbol} closed successfully'})
    else:
        return jsonify({'error': f'Failed to close position for {symbol}'})

if __name__ == '__main__':
    # Initialize bot on startup
    print("🚀 Starting Flask Web Interface...")
    initialize_bot()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)