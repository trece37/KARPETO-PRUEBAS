from datetime import datetime
from typing import List
from ..core.interfaces import RiskManagementModel
from ..core.types import PortfolioTarget

class CircuitBreaker(RiskManagementModel):
    def __init__(self, max_daily_loss_percent=0.03):
        self.max_daily_loss_percent = max_daily_loss_percent
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        self.triggered = False

    def manage_risk(self, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        """
        Implementation of the RiskManagementModel interface.
        If circuit breaker is triggered, force all targets towards zero (Liquidate).
        """
        # Note: In a real QC model, we'd need access to the algorithm state to check drawdown
        # For this implementation, we assume external state injection or update via check_safety
        
        if self.triggered:
            # Liquidate everything
            return [PortfolioTarget(symbol=t.symbol, quantity=0, percent=0.0) for t in targets]
            
        return targets

    def update_pnl(self, realized_pnl):
        self._check_reset()
        self.daily_pnl += realized_pnl
        # Check logic here (needs balance context usually, simplified for now)
        pass

    def check_safety(self, current_drawdown_percent):
        """
        Returns False if trading should stop.
        """
        self._check_reset()
        
        if self.triggered:
            return False, "Circuit Breaker previously triggered today."

        if current_drawdown_percent >= self.max_daily_loss_percent:
            self.triggered = True
            return False, f"Circuit Breaker TRIGGERED: Drawdown {current_drawdown_percent:.2%} > Limit {self.max_daily_loss_percent:.2%}"
            
        return True, "Safe"

    def _check_reset(self):
        if datetime.now().date() > self.last_reset:
            self.daily_pnl = 0.0
            self.triggered = False
            self.last_reset = datetime.now().date()
