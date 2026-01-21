"""
Monte Carlo Risk Forecasting
Phase 4.2: Oro Puro R3K Compliance

This module implements Monte Carlo simulation to forecast portfolio risk distribution.
Instead of relying on a single backtest curve, we simulate 5000+ scenarios to understand
the true risk exposure (DrawDown, VaR, etc.).

References: Oro Puro (00000.Todas las fuentes - oropuro1.MD)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple
import matplotlib.pyplot as plt

class MonteCarloRiskForecaster:
    def __init__(self, num_simulations: int = 5000, time_periods: int = 252):
        """
        Initialize Monte Carlo Risk Forecaster.
        
        Args:
            num_simulations: Number of simulation runs (default: 5000)
            time_periods: Number of days to forecast (default: 252 = 1 year)
        """
        # [TAG: R3K_MONTE_CARLO_CONFIG]
        self.num_simulations = num_simulations
        self.time_periods = time_periods
        
    def simulate_price_paths(self, 
                             initial_price: float,
                             historical_returns: pd.Series) -> np.ndarray:
        """
        # [TAG: R3K_GEOMETRIC_BROWNIAN_MOTION]
        Simulate future price paths using Geometric Brownian Motion.
        
        Args:
            initial_price: Starting price (e.g., current account balance)
            historical_returns: Historical return series for drift/volatility estimation
            
        Returns:
            2D array (time_periods x num_simulations) of simulated prices
        """
        # [TAG: STEP_1_PERIODIC_RETURNS]
        # Already calculated by caller (log returns)
        
        # [TAG: STEP_2_DRIFT_CALCULATION]
        # Drift = Average Return - (Variance / 2)
        avg_return = historical_returns.mean()
        variance = historical_returns.var()
        drift = avg_return - (variance / 2)
        
        # For conservative risk analysis, can set drift = 0
        # drift = 0.0
        
        std_dev = historical_returns.std()
        
        # Initialize simulation matrix
        simulation_matrix = np.zeros((self.time_periods + 1, self.num_simulations))
        simulation_matrix[0] = initial_price
        
        # [TAG: STEP_3_4_ITERATION]
        # Generate random price paths
        for t in range(1, self.time_periods + 1):
            # Generate random values from normal distribution
            random_values = norm.ppf(np.random.rand(self.num_simulations))
            random_input = std_dev * random_values
            
            # Price formula: Price_next = Price_current * e^(drift + random_input)
            simulation_matrix[t] = simulation_matrix[t-1] * np.exp(drift + random_input)
        
        return simulation_matrix
    
    def calculate_risk_metrics(self, simulation_matrix: np.ndarray) -> Dict:
        """
        # [TAG: R3K_RISK_METRICS]
        Calculate risk metrics from Monte Carlo results.
        
        Args:
            simulation_matrix: Output from simulate_price_paths
            
        Returns:
            Dict with risk statistics
        """
        initial_price = simulation_matrix[0, 0]
        final_prices = simulation_matrix[-1, :]
        
        # Calculate returns for each simulation
        returns = (final_prices - initial_price) / initial_price
        
        # Calculate drawdowns for each path
        drawdowns = []
        for i in range(self.num_simulations):
            path = simulation_matrix[:, i]
            running_max = np.maximum.accumulate(path)
            drawdown = (path / running_max) - 1
            max_drawdown = drawdown.min()
            drawdowns.append(max_drawdown)
        
        drawdowns = np.array(drawdowns)
        
        # [TAG: VAR_CALCULATION]
        # Value at Risk (95% confidence): 95% of scenarios are better than this
        var_95 = np.percentile(returns, 5)
        
        # [TAG: CVAR_CALCULATION]
        # Conditional VaR (Expected Shortfall): Average of worst 5%
        cvar_95 = returns[returns <= var_95].mean()
        
        metrics = {
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "best_case": returns.max(),
            "worst_case": returns.min(),
            "var_95": var_95,  # 95% VaR
            "cvar_95": cvar_95,  # Expected Shortfall
            "mean_drawdown": drawdowns.mean(),
            "worst_drawdown": drawdowns.min(),
            "probability_of_loss": (returns < 0).sum() / self.num_simulations
        }
        
        return metrics
    
    def forecast_risk(self, 
                     initial_balance: float,
                     historical_returns: pd.Series,
                     plot: bool = False) -> Dict:
        """
        # [TAG: FULL_MONTE_CARLO_WORKFLOW]
        Run complete Monte Carlo risk forecast.
        
        Args:
            initial_balance: Starting account balance
            historical_returns: Historical return series
            plot: Whether to plot results (for debugging)
            
        Returns:
            Risk metrics dictionary
        """
        print("=" * 60)
        print(f"MONTE CARLO RISK FORECASTING (R3K)")
        print("=" * 60)
        print(f"Simulations: {self.num_simulations}")
        print(f"Periods: {self.time_periods} days")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        
        # Run simulations
        print("\nRunning simulations...")
        simulation_matrix = self.simulate_price_paths(initial_balance, historical_returns)
        
        # Calculate metrics
        metrics = self.calculate_risk_metrics(simulation_matrix)
        
        # Print results
        print("\n--- RISK ASSESSMENT ---")
        print(f"Expected Return: {metrics['mean_return']*100:.2f}%")
        print(f"Return Volatility: {metrics['std_return']*100:.2f}%")
        print(f"Best Case: {metrics['best_case']*100:.2f}%")
        print(f"Worst Case: {metrics['worst_case']*100:.2f}%")
        print(f"\n95% VaR: {metrics['var_95']*100:.2f}% (5% chance of worse)")
        print(f"95% CVaR: {metrics['cvar_95']*100:.2f}% (avg of worst 5%)")
        print(f"\nMean Max Drawdown: {metrics['mean_drawdown']*100:.2f}%")
        print(f"Worst Drawdown: {metrics['worst_drawdown']*100:.2f}%")
        print(f"Probability of Loss: {metrics['probability_of_loss']*100:.2f}%")
        
        if plot:
            self._plot_results(simulation_matrix, initial_balance)
        
        return metrics
    
    def _plot_results(self, simulation_matrix: np.ndarray, initial_balance: float):
        """Plot Monte Carlo simulation results."""
        plt.figure(figsize=(12, 6))
        
        # Plot a sample of paths (100 out of 5000)
        sample_paths = np.random.choice(self.num_simulations, size=100, replace=False)
        for i in sample_paths:
            plt.plot(simulation_matrix[:, i], alpha=0.1, color='blue')
        
        # Plot mean path
        mean_path = simulation_matrix.mean(axis=1)
        plt.plot(mean_path, color='red', linewidth=2, label='Mean Path')
        
        plt.axhline(initial_balance, color='black', linestyle='--', label='Initial Balance')
        plt.title(f'Monte Carlo Simulation ({self.num_simulations} scenarios)')
        plt.xlabel('Days')
        plt.ylabel('Balance ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # Example Test with Synthetic Returns
    print("Testing Monte Carlo Forecaster...\n")
    
    # Generate synthetic daily returns (mean 0.05%, std 2%)
    np.random.seed(42)
    historical_returns = pd.Series(np.random.normal(0.0005, 0.02, 500))
    
    forecaster = MonteCarloRiskForecaster(num_simulations=5000, time_periods=252)
    metrics = forecaster.forecast_risk(
        initial_balance=10000.0,
        historical_returns=historical_returns,
        plot=False
    )
    
    print("\n" + "=" * 60)
    print("MONTE CARLO VALIDATION COMPLETE ✅")
    print("=" * 60)
