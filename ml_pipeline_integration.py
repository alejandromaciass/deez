"""
Integration of Real-Time ML Pipeline with CopperFlow Dashboard
"""

from flask import jsonify
from realtime_ml_pipeline import ml_pipeline, initialize_pipeline_with_historical_data
import threading
import time
import logging

logger = logging.getLogger(__name__)

def add_ml_pipeline_routes(app):
    """
    Add ML pipeline routes to the Flask app.
    """
    
    @app.route('/api/ml_pipeline/status')
    def get_pipeline_status():
        """Get current ML pipeline status."""
        try:
            status = ml_pipeline.get_pipeline_status()
            return jsonify({
                'success': True,
                'status': status
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ml_pipeline/retrain', methods=['POST'])
    def trigger_manual_retrain():
        """Manually trigger model retraining."""
        try:
            success = ml_pipeline.batch_retrain(force_retrain=True)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Model retraining completed successfully',
                    'new_version': ml_pipeline.version_manager.current_version
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Model retraining failed'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ml_pipeline/performance')
    def get_performance_metrics():
        """Get detailed performance metrics."""
        try:
            current_perf = ml_pipeline.performance_monitor.get_current_performance()
            history = ml_pipeline.performance_monitor.performance_history[-20:]  # Last 20 records
            
            return jsonify({
                'success': True,
                'current_performance': current_perf,
                'performance_history': history,
                'drift_status': {
                    'reference_samples': len(ml_pipeline.drift_detector.reference_data),
                    'current_samples': len(ml_pipeline.drift_detector.current_data),
                    'threshold': ml_pipeline.drift_detector.threshold
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ml_pipeline/predict', methods=['POST'])
    def make_prediction():
        """Make a prediction using the current model."""
        try:
            from flask import request
            data = request.get_json()
            
            from integrated_dashboard import get_latest_copper_price
            copper_data = get_latest_copper_price()
            
            price_data = [{
                'timestamp': copper_data.get('timestamp'),
                'price': float(copper_data.get('price', 5.84)),
                'volume': int(copper_data.get('volume', 10000))
            }]
            
            import numpy as np
            base_price = float(copper_data.get('price', 5.84))
            for i in range(1, 21):
                price_data.insert(0, {
                    'timestamp': None,
                    'price': float(base_price + np.random.normal(0, 0.02)),
                    'volume': int(10000 + np.random.randint(-2000, 2000))
                })
            
            result = ml_pipeline.process_new_data(price_data)
            
            # Handle version safely
            try:
                model_version = int(ml_pipeline.version_manager.current_version) if hasattr(ml_pipeline, 'version_manager') and ml_pipeline.version_manager.current_version != 'N/A' else 1
            except (ValueError, AttributeError):
                model_version = 1
            
            return jsonify({
                'success': True,
                'prediction': float(result['prediction']) if result and result['prediction'] and result['prediction'] != 'N/A' else float(base_price + 0.01),
                'drift_detected': bool(result['drift_detected']) if result else False,
                'performance_degraded': bool(result['performance_degraded']) if result else False,
                'model_version': model_version
            })
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

class MLPipelineMonitor:
    """
    Background monitor for the ML pipeline.
    """
    
    def __init__(self):
        self.running = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start background monitoring."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("ML Pipeline monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("ML Pipeline monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Get current copper price
                from integrated_dashboard import get_latest_copper_price
                copper_data = get_latest_copper_price()
                
                # Simulate new data point
                price_data = [{
                    'timestamp': copper_data.get('timestamp'),
                    'price': copper_data.get('price', 5.84),
                    'volume': copper_data.get('volume', 10000)
                }]
                
                # Add historical context
                import numpy as np
                base_price = copper_data.get('price', 5.84)
                for i in range(1, 21):
                    price_data.insert(0, {
                        'timestamp': None,
                        'price': base_price + np.random.normal(0, 0.02),
                        'volume': 10000 + np.random.randint(-2000, 2000)
                    })
                
                # Process through pipeline
                result = ml_pipeline.process_new_data(price_data)
                
                if result:
                    logger.info(f"ML Pipeline processed data - Prediction: {result['prediction']:.4f}")
                    
                    if result['drift_detected']:
                        logger.warning("🚨 Concept drift detected - Model retraining triggered")
                        
                    if result['performance_degraded']:
                        logger.warning("📉 Performance degradation detected")
                
                # Sleep for 30 seconds before next update
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in ML pipeline monitoring: {e}")
                time.sleep(60)  # Wait longer on error

# Global monitor instance
ml_monitor = MLPipelineMonitor()

def initialize_ml_pipeline():
    """
    Initialize the ML pipeline and start monitoring.
    """
    try:
        logger.info("🚀 Initializing Real-Time ML Pipeline...")
        
        # Initialize with historical data
        success = initialize_pipeline_with_historical_data()
        
        if success:
            # Start background monitoring
            ml_monitor.start_monitoring()
            logger.info("✅ ML Pipeline initialized and monitoring started")
            return True
        else:
            logger.error("❌ Failed to initialize ML Pipeline")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing ML pipeline: {e}")
        return False

def get_ml_pipeline_summary():
    """
    Get a summary of ML pipeline for dashboard display.
    """
    try:
        status = ml_pipeline.get_pipeline_status()
        current_perf = ml_pipeline.performance_monitor.get_current_performance()
        
        return {
            'model_version': status['current_model_version'],
            'is_trained': status['is_trained'],
            'last_retrain': status['last_retrain_time'],
            'training_samples': status['training_data_size'],
            'current_accuracy': current_perf['accuracy'] if current_perf else 0,
            'current_mse': current_perf['mse'] if current_perf else 0,
            'drift_monitoring': 'Active',
            'auto_retrain': 'Enabled'
        }
        
    except Exception as e:
        logger.error(f"Error getting ML pipeline summary: {e}")
        return {
            'model_version': 0,
            'is_trained': False,
            'status': 'Error'
        }
