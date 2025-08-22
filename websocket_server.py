"""
WebSocket Server for Real-Time CopperFlow Analytics
"""

from flask_socketio import SocketIO, emit
import threading
import time
import json
from datetime import datetime
import numpy as np

def setup_websocket(app):
    """
    Setup WebSocket server for real-time data streaming.
    """
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Global variables for real-time data
    connected_clients = set()
    data_thread = None
    thread_lock = threading.Lock()
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        connected_clients.add(request.sid)
        print(f"Client {request.sid} connected. Total clients: {len(connected_clients)}")
        
        # Send initial data to new client
        try:
            from integrated_dashboard import get_latest_copper_price
            initial_data = get_latest_copper_price()
            emit('price_update', initial_data)
            emit('connection_status', {'status': 'connected', 'timestamp': datetime.now().isoformat()})
        except Exception as e:
            print(f"Error sending initial data: {e}")
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        connected_clients.discard(request.sid)
        print(f"Client {request.sid} disconnected. Total clients: {len(connected_clients)}")
    
    @socketio.on('request_update')
    def handle_update_request():
        """Handle manual update request from client."""
        try:
            from integrated_dashboard import get_latest_copper_price
            current_data = get_latest_copper_price()
            emit('price_update', current_data)
            print(f"Manual update sent to client {request.sid}")
        except Exception as e:
            print(f"Error handling update request: {e}")
            emit('error', {'message': 'Failed to fetch latest data'})
    
    @socketio.on('subscribe_alerts')
    def handle_alert_subscription(data):
        """Handle price alert subscriptions."""
        threshold = data.get('threshold', 0)
        alert_type = data.get('type', 'price_change')  # price_change, volume_spike, etc.
        
        # Store alert subscription (in production, use database)
        client_alerts = {
            'client_id': request.sid,
            'threshold': threshold,
            'type': alert_type,
            'created_at': datetime.now().isoformat()
        }
        
        emit('alert_subscribed', {
            'status': 'success',
            'alert': client_alerts
        })
        print(f"Alert subscription created for client {request.sid}: {client_alerts}")
    
    def background_data_stream():
        """
        Background thread to stream real-time data to all connected clients.
        """
        print("Starting background data stream...")
        
        while True:
            try:
                if connected_clients:
                    # Get latest market data
                    from integrated_dashboard import get_latest_copper_price
                    current_data = get_latest_copper_price()
                    
                    # Add timestamp for real-time tracking
                    current_data['stream_timestamp'] = datetime.now().isoformat()
                    current_data['clients_connected'] = len(connected_clients)
                    
                    # Broadcast to all connected clients
                    socketio.emit('price_update', current_data)
                    
                    # Check for alerts (simplified example)
                    if 'price' in current_data:
                        price = current_data['price']
                        change_percent = current_data.get('change_percent', 0)
                        
                        # Trigger alerts for significant price movements
                        if abs(change_percent) > 1.0:  # 1% change threshold
                            alert_data = {
                                'type': 'price_alert',
                                'message': f"Significant price movement: {change_percent:+.2f}%",
                                'price': price,
                                'change_percent': change_percent,
                                'timestamp': datetime.now().isoformat()
                            }
                            socketio.emit('trading_alert', alert_data)
                    
                    print(f"Data streamed to {len(connected_clients)} clients at {datetime.now()}")
                
                # Wait 5 seconds before next update
                time.sleep(5)
                
            except Exception as e:
                print(f"Error in background data stream: {e}")
                time.sleep(10)  # Wait longer on error
    
    def start_background_thread():
        """Start the background data streaming thread."""
        global data_thread
        with thread_lock:
            if data_thread is None:
                data_thread = threading.Thread(target=background_data_stream)
                data_thread.daemon = True
                data_thread.start()
    
    # Start background thread when first client connects
    @socketio.on('connect')
    def start_stream_on_connect():
        start_background_thread()
    
    # Health check endpoint
    @app.route('/api/websocket/status')
    def websocket_status():
        return {
            'status': 'active',
            'connected_clients': len(connected_clients),
            'thread_active': data_thread is not None and data_thread.is_alive(),
            'timestamp': datetime.now().isoformat()
        }
    
    return socketio

# Usage in integrated_dashboard.py
"""
Add this to your integrated_dashboard.py:

from websocket_server import setup_websocket

# After creating the Flask app
socketio = setup_websocket(app)

# Change the app.run() to:
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=8082, debug=True)
"""
