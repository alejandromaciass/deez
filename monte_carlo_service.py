"""
Monte Carlo Simulation Service for CopperFlow Analytics
"""

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulation for commodity price forecasting.
    """
    
    def __init__(self):
        self.historical_data = {}
        self.simulation_cache = {}
    
    def fetch_historical_data(self, symbol='HG=F', period='1y'):
        """
        Fetch historical price data for Monte Carlo simulation.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change().dropna()
            
            self.historical_data[symbol] = data
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def calculate_volatility_parameters(self, symbol='HG=F'):
        """
        Calculate volatility parameters for Monte Carlo simulation.
        """
        if symbol not in self.historical_data:
            self.fetch_historical_data(symbol)
        
        data = self.historical_data.get(symbol)
        if data is None or data.empty:
            # Use default parameters if no data available
            return {
                'mean_return': 0.0005,  # 0.05% daily return
                'volatility': 0.025,    # 2.5% daily volatility
                'drift': 0.0002
            }
        
        returns = data['Returns'].dropna()
        
        # Calculate parameters
        mean_return = returns.mean()
        volatility = returns.std()
        drift = mean_return - (volatility ** 2) / 2
        
        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'drift': drift,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
    
    def run_monte_carlo_simulation(self, 
                                 current_price=5.84,
                                 days=252,  # 1 year
                                 n_simulations=1000,
                                 symbol='HG=F'):
        """
        Run Monte Carlo simulation for price forecasting.
        """
        logger.info(f"Running Monte Carlo simulation: {n_simulations} paths, {days} days")
        
        # Get volatility parameters
        params = self.calculate_volatility_parameters(symbol)
        
        # Initialize arrays
        dt = 1/252  # Daily time step
        price_paths = np.zeros((n_simulations, days + 1))
        price_paths[:, 0] = current_price
        
        # Generate random shocks
        np.random.seed(42)  # For reproducible results
        random_shocks = np.random.normal(0, 1, (n_simulations, days))
        
        # Simulate price paths using Geometric Brownian Motion
        for t in range(1, days + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (params['drift']) * dt + 
                params['volatility'] * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        
        results = {
            'simulation_params': {
                'current_price': current_price,
                'days': days,
                'n_simulations': n_simulations,
                'mean_return': params['mean_return'],
                'volatility': params['volatility'],
                'drift': params['drift']
            },
            'price_paths': price_paths.tolist(),
            'final_prices': final_prices.tolist(),
            'statistics': {
                'mean_final_price': float(np.mean(final_prices)),
                'median_final_price': float(np.median(final_prices)),
                'std_final_price': float(np.std(final_prices)),
                'min_price': float(np.min(final_prices)),
                'max_price': float(np.max(final_prices)),
                'var_95': float((current_price - np.percentile(final_prices, 5)) / current_price),  # VaR as percentage loss
                'var_99': float((current_price - np.percentile(final_prices, 1)) / current_price),  # VaR as percentage loss
                'probability_profit': float(np.mean(final_prices > current_price)),
                'expected_return': float((np.mean(final_prices) - current_price) / current_price),
                'percentiles': {
                    '10th': float(np.percentile(final_prices, 10)),
                    '25th': float(np.percentile(final_prices, 25)),
                    '75th': float(np.percentile(final_prices, 75)),
                    '90th': float(np.percentile(final_prices, 90))
                }
            },
            'risk_metrics': self.calculate_risk_metrics(price_paths, current_price),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache results
        cache_key = f"{symbol}_{current_price}_{days}_{n_simulations}"
        self.simulation_cache[cache_key] = results
        
        return results
    
    def calculate_risk_metrics(self, price_paths, initial_price):
        """
        Calculate comprehensive risk metrics from simulation results.
        """
        # Calculate returns for each path
        returns = (price_paths[:, -1] - initial_price) / initial_price
        
        # Maximum drawdown calculation
        max_drawdowns = []
        for path in price_paths:
            running_max = np.maximum.accumulate(path)
            drawdowns = (path - running_max) / running_max
            max_drawdowns.append(np.min(drawdowns))
        
        return {
            'value_at_risk': {
                '95%': float(np.percentile(returns, 5)),
                '99%': float(np.percentile(returns, 1)),
                '99.9%': float(np.percentile(returns, 0.1))
            },
            'conditional_var': {
                '95%': float(np.mean(returns[returns <= np.percentile(returns, 5)])),
                '99%': float(np.mean(returns[returns <= np.percentile(returns, 1)]))
            },
            'maximum_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'worst': float(np.min(max_drawdowns)),
                '95th_percentile': float(np.percentile(max_drawdowns, 5))
            },
            'sharpe_ratio': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'probability_of_loss': float(np.mean(returns < 0))
        }
    
    def calculate_sortino_ratio(self, returns, target_return=0):
        """
        Calculate Sortino ratio (downside deviation).
        """
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return float('inf')
        
        return float((np.mean(returns) - target_return) / downside_deviation)
    
    def get_simulation_summary(self, results):
        """
        Generate a human-readable summary of simulation results.
        """
        stats = results['statistics']
        risk = results['risk_metrics']
        
        summary = {
            'title': 'Monte Carlo Price Simulation Results',
            'key_insights': [
                f"Expected price in {results['simulation_params']['days']} days: ${stats['mean_final_price']:.2f}",
                f"Probability of profit: {stats['probability_profit']:.1%}",
                f"95% Value at Risk: {risk['value_at_risk']['95%']:.1%}",
                f"Maximum expected drawdown: {risk['maximum_drawdown']['worst']:.1%}"
            ],
            'price_range': {
                'optimistic': stats['percentiles']['90th'],
                'expected': stats['mean_final_price'],
                'pessimistic': stats['percentiles']['10th']
            },
            'risk_assessment': self.assess_risk_level(risk),
            'recommendation': self.generate_recommendation(stats, risk)
        }
        
        return summary
    
    def assess_risk_level(self, risk_metrics):
        """
        Assess overall risk level based on metrics.
        """
        var_95 = abs(risk_metrics['value_at_risk']['95%'])
        max_dd = abs(risk_metrics['maximum_drawdown']['worst'])
        
        if var_95 > 0.2 or max_dd > 0.3:
            return 'High Risk'
        elif var_95 > 0.1 or max_dd > 0.15:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    def generate_recommendation(self, stats, risk_metrics):
        """
        Generate trading recommendation based on simulation results.
        """
        prob_profit = stats['probability_profit']
        expected_return = stats['expected_return']
        sharpe_ratio = risk_metrics['sharpe_ratio']
        
        if prob_profit > 0.6 and expected_return > 0.05 and sharpe_ratio > 0.5:
            return 'Strong Buy - Favorable risk/reward profile'
        elif prob_profit > 0.55 and expected_return > 0.02:
            return 'Buy - Positive expected return with acceptable risk'
        elif prob_profit < 0.4 or expected_return < -0.05:
            return 'Sell - Unfavorable risk/reward profile'
        else:
            return 'Hold - Neutral outlook with balanced risk/reward'

# Global simulator instance
monte_carlo_simulator = MonteCarloSimulator()

def run_simulation_api(current_price=5.84, days=252, n_simulations=1000):
    """
    API endpoint function for running Monte Carlo simulation.
    """
    try:
        results = monte_carlo_simulator.run_monte_carlo_simulation(
            current_price=current_price,
            days=days,
            n_simulations=n_simulations
        )
        
        summary = monte_carlo_simulator.get_simulation_summary(results)
        
        return {
            'success': True,
            'results': results,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {e}")
        return {
            'success': False,
            'error': str(e)
        }
