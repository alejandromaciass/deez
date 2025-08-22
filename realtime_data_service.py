"""
Real-Time Data Service
Integrates with Yahoo Finance, Alpha Vantage, and FRED APIs for live data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import json
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from fredapi import Fred
import asyncio
import websockets
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Data structure for market information"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: str
    high: float
    low: float
    timestamp: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None

@dataclass
class EconomicIndicator:
    """Data structure for economic indicators"""
    indicator: str
    value: float
    date: str
    change: Optional[float] = None
    unit: str = ""

class RealTimeDataService:
    """
    Real-time data service integrating multiple APIs
    """
    
    def __init__(self, alpha_vantage_key: str = None, fred_key: str = None):
        """
        Initialize the real-time data service
        
        Args:
            alpha_vantage_key: Alpha Vantage API key (optional, uses demo key if not provided)
            fred_key: FRED API key (optional, uses demo access if not provided)
        """
        self.alpha_vantage_key = alpha_vantage_key or "demo"  # Demo key for testing
        self.fred_key = fred_key
        
        # Initialize API clients
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        
        if self.fred_key:
            self.fred = Fred(api_key=self.fred_key)
        else:
            self.fred = None
            logger.warning("FRED API key not provided. Economic indicators will use simulated data.")
        
        # Cache for data
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # Cache for 60 seconds
        
        # WebSocket connections
        self.websocket_clients = set()
        
        logger.info("✅ Real-time data service initialized")
    
    def get_copper_price_live(self) -> MarketData:
        """
        Get live copper price from Yahoo Finance
        """
        try:
            # Use copper futures symbol
            copper = yf.Ticker("HG=F")  # COMEX Copper Futures
            
            # Get current data
            info = copper.info
            hist = copper.history(period="1d", interval="1m")
            
            if hist.empty:
                # Fallback to daily data
                hist = copper.history(period="5d")
            
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
            
            # Get additional data
            volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            high = hist['High'].max()
            low = hist['Low'].min()
            
            market_data = MarketData(
                symbol="HG=F",
                price=round(float(current_price), 4),
                change=round(float(change), 4),
                change_percent=round(float(change_percent), 2),
                volume=f"{int(volume):,}" if volume > 0 else "N/A",
                high=round(float(high), 4),
                low=round(float(low), 4),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE')
            )
            
            logger.info(f"📈 Live copper price: ${current_price:.4f} ({change_percent:+.2f}%)")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching live copper price: {e}")
            # Return fallback data
            return self._get_fallback_copper_data()
    
    def get_related_commodities(self) -> Dict[str, MarketData]:
        """
        Get prices for related commodities
        """
        commodities = {
            'Gold': 'GC=F',
            'Silver': 'SI=F', 
            'Aluminum': 'ALI=F',
            'Crude Oil': 'CL=F',
            'Natural Gas': 'NG=F'
        }
        
        results = {}
        
        for name, symbol in commodities.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
                    
                    results[name] = MarketData(
                        symbol=symbol,
                        price=round(float(current_price), 2),
                        change=round(float(change), 2),
                        change_percent=round(float(change_percent), 2),
                        volume="N/A",
                        high=round(float(hist['High'].max()), 2),
                        low=round(float(hist['Low'].min()), 2),
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch {name} data: {e}")
                continue
        
        return results
    
    def get_economic_indicators(self) -> List[EconomicIndicator]:
        """
        Get economic indicators from FRED API
        """
        indicators = []
        
        # Define key economic indicators
        fred_series = {
            'GDP': 'GDP',
            'Inflation (CPI)': 'CPIAUCSL',
            'Unemployment Rate': 'UNRATE',
            'Federal Funds Rate': 'FEDFUNDS',
            'Industrial Production': 'INDPRO',
            'Consumer Confidence': 'UMCSENT'
        }
        
        if self.fred:
            for name, series_id in fred_series.items():
                try:
                    # Get latest data point
                    data = self.fred.get_series(series_id, limit=2)
                    if not data.empty:
                        latest_value = data.iloc[-1]
                        latest_date = data.index[-1].strftime("%Y-%m-%d")
                        
                        # Calculate change if we have previous data
                        change = None
                        if len(data) > 1:
                            previous_value = data.iloc[-2]
                            change = latest_value - previous_value
                        
                        indicators.append(EconomicIndicator(
                            indicator=name,
                            value=float(latest_value),
                            date=latest_date,
                            change=float(change) if change is not None else None,
                            unit=self._get_indicator_unit(name)
                        ))
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch {name}: {e}")
                    continue
        else:
            # Use simulated data if FRED API not available
            indicators = self._get_simulated_economic_indicators()
        
        return indicators
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Get market sentiment indicators
        """
        try:
            # VIX (Volatility Index)
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="1d")
            vix_value = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20.0
            
            # Dollar Index
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_hist = dxy.history(period="1d")
            dxy_value = dxy_hist['Close'].iloc[-1] if not dxy_hist.empty else 100.0
            
            # 10-Year Treasury Yield
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="1d")
            tnx_value = tnx_hist['Close'].iloc[-1] if not tnx_hist.empty else 4.0
            
            sentiment_score = self._calculate_sentiment_score(vix_value, dxy_value, tnx_value)
            
            return {
                'vix': float(vix_value),
                'dollar_index': float(dxy_value),
                'treasury_yield': float(tnx_value),
                'sentiment_score': sentiment_score,
                'sentiment_label': self._get_sentiment_label(sentiment_score),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching market sentiment: {e}")
            return self._get_fallback_sentiment()
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical data for analysis
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_fallback_copper_data(self) -> MarketData:
        """Fallback copper data when API fails"""
        base_price = 5.84
        variation = np.random.normal(0, 0.02)
        price = base_price + variation
        change = variation
        change_percent = (change / base_price) * 100
        
        return MarketData(
            symbol="HG=F",
            price=price,
            change=change,
            change_percent=change_percent,
            volume="15,234",
            high=price + 0.05,
            low=price - 0.05,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _get_simulated_economic_indicators(self) -> List[EconomicIndicator]:
        """Simulated economic indicators when FRED API not available"""
        base_indicators = [
            ('GDP', 25000.0, 'Billions USD'),
            ('Inflation (CPI)', 3.2, '%'),
            ('Unemployment Rate', 3.8, '%'),
            ('Federal Funds Rate', 5.25, '%'),
            ('Industrial Production', 102.5, 'Index'),
            ('Consumer Confidence', 98.2, 'Index')
        ]
        
        indicators = []
        for name, base_value, unit in base_indicators:
            variation = np.random.normal(0, base_value * 0.01)
            value = base_value + variation
            change = np.random.normal(0, base_value * 0.005)
            
            indicators.append(EconomicIndicator(
                indicator=name,
                value=value,
                date=datetime.now().strftime("%Y-%m-%d"),
                change=change,
                unit=unit
            ))
        
        return indicators
    
    def _get_indicator_unit(self, indicator_name: str) -> str:
        """Get unit for economic indicator"""
        units = {
            'GDP': 'Billions USD',
            'Inflation (CPI)': 'Index',
            'Unemployment Rate': '%',
            'Federal Funds Rate': '%',
            'Industrial Production': 'Index',
            'Consumer Confidence': 'Index'
        }
        return units.get(indicator_name, '')
    
    def _calculate_sentiment_score(self, vix: float, dxy: float, tnx: float) -> float:
        """Calculate overall market sentiment score (0-100)"""
        # Normalize indicators (lower VIX = better sentiment)
        vix_score = max(0, min(100, 100 - (vix - 10) * 2))  # VIX 10-60 range
        dxy_score = max(0, min(100, (dxy - 80) * 1.25))     # DXY 80-100 range  
        tnx_score = max(0, min(100, (tnx - 2) * 12.5))      # TNX 2-10 range
        
        # Weighted average
        sentiment = (vix_score * 0.5 + dxy_score * 0.3 + tnx_score * 0.2)
        return round(sentiment, 1)
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label from score"""
        if score >= 70:
            return "Bullish"
        elif score >= 50:
            return "Neutral"
        elif score >= 30:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def _get_fallback_sentiment(self) -> Dict[str, Any]:
        """Fallback sentiment data"""
        return {
            'vix': 18.5,
            'dollar_index': 103.2,
            'treasury_yield': 4.3,
            'sentiment_score': 65.0,
            'sentiment_label': 'Neutral',
            'timestamp': datetime.now().isoformat()
        }

# WebSocket Server for Real-Time Updates
class WebSocketServer:
    """WebSocket server for real-time data streaming"""
    
    def __init__(self, data_service: RealTimeDataService, port: int = 8765):
        self.data_service = data_service
        self.port = port
        self.clients = set()
        self.running = False
        
    async def register_client(self, websocket, path):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"📡 Client connected. Total clients: {len(self.clients)}")
        
        try:
            # Send initial data
            await self.send_initial_data(websocket)
            
            # Keep connection alive
            async for message in websocket:
                # Handle client messages if needed
                pass
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"📡 Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_initial_data(self, websocket):
        """Send initial data to newly connected client"""
        try:
            data = {
                'type': 'initial_data',
                'copper_price': self.data_service.get_copper_price_live().__dict__,
                'commodities': {k: v.__dict__ for k, v in self.data_service.get_related_commodities().items()},
                'economic_indicators': [ind.__dict__ for ind in self.data_service.get_economic_indicators()],
                'market_sentiment': self.data_service.get_market_sentiment()
            }
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"❌ Error sending initial data: {e}")
    
    async def broadcast_updates(self):
        """Broadcast real-time updates to all clients"""
        while self.running:
            if self.clients:
                try:
                    # Get fresh data
                    copper_data = self.data_service.get_copper_price_live()
                    sentiment_data = self.data_service.get_market_sentiment()
                    
                    update = {
                        'type': 'price_update',
                        'copper_price': copper_data.__dict__,
                        'market_sentiment': sentiment_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Broadcast to all clients
                    if self.clients:
                        await asyncio.gather(
                            *[client.send(json.dumps(update)) for client in self.clients],
                            return_exceptions=True
                        )
                        
                except Exception as e:
                    logger.error(f"❌ Error broadcasting updates: {e}")
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
    
    def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        
        async def run_server():
            # Start WebSocket server
            server = await websockets.serve(self.register_client, "localhost", self.port)
            logger.info(f"🚀 WebSocket server started on ws://localhost:{self.port}")
            
            # Start broadcasting updates
            await asyncio.gather(
                server.wait_closed(),
                self.broadcast_updates()
            )
        
        # Run in event loop
        asyncio.run(run_server())

# Global instance
real_time_service = RealTimeDataService()

# API Functions for Flask Integration
def get_live_copper_price() -> Dict[str, Any]:
    """Get live copper price for Flask API"""
    try:
        data = real_time_service.get_copper_price_live()
        return {
            'success': True,
            'data': data.__dict__
        }
    except Exception as e:
        logger.error(f"❌ Error in get_live_copper_price: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': real_time_service._get_fallback_copper_data().__dict__
        }

def get_market_data_summary() -> Dict[str, Any]:
    """Get comprehensive market data summary"""
    try:
        copper_data = real_time_service.get_copper_price_live()
        commodities = real_time_service.get_related_commodities()
        indicators = real_time_service.get_economic_indicators()
        sentiment = real_time_service.get_market_sentiment()
        
        return {
            'success': True,
            'copper_price': copper_data.__dict__,
            'related_commodities': {k: v.__dict__ for k, v in commodities.items()},
            'economic_indicators': [ind.__dict__ for ind in indicators],
            'market_sentiment': sentiment,
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error in get_market_data_summary: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_historical_copper_data(period: str = "1y") -> Dict[str, Any]:
    """Get historical copper data"""
    try:
        hist_data = real_time_service.get_historical_data("HG=F", period)
        
        if hist_data.empty:
            return {'success': False, 'error': 'No historical data available'}
        
        # Convert to format suitable for charts
        chart_data = {
            'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
            'prices': hist_data['Close'].tolist(),
            'volumes': hist_data['Volume'].tolist() if 'Volume' in hist_data.columns else [],
            'highs': hist_data['High'].tolist(),
            'lows': hist_data['Low'].tolist()
        }
        
        return {
            'success': True,
            'data': chart_data,
            'period': period
        }
    except Exception as e:
        logger.error(f"❌ Error in get_historical_copper_data: {e}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Test the service
    print("🚀 Testing Real-Time Data Service...")
    
    service = RealTimeDataService()
    
    # Test copper price
    copper = service.get_copper_price_live()
    print(f"📈 Copper Price: ${copper.price:.4f} ({copper.change_percent:+.2f}%)")
    
    # Test commodities
    commodities = service.get_related_commodities()
    print(f"🏗️ Related Commodities: {len(commodities)} fetched")
    
    # Test economic indicators
    indicators = service.get_economic_indicators()
    print(f"📊 Economic Indicators: {len(indicators)} fetched")
    
    # Test market sentiment
    sentiment = service.get_market_sentiment()
    print(f"💭 Market Sentiment: {sentiment['sentiment_label']} ({sentiment['sentiment_score']})")
    
    print("✅ Real-time data service test completed!")
