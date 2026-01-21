from datetime import datetime
from typing import Dict, Any, Tuple
from .state_manager import StateManager

class RiskEngine:
    """
    Achilles Risk Engine (The Shield).
    Enforces "4RULES" compliance:
    1. Daily Drawdown Limit.
    2. Global Kill Switch.
    """
    def __init__(self, state_manager: StateManager):
        self.state = state_manager
        
        # [TAG: R3K_RISK_PARAMETERS]
        self.MAX_DAILY_DRAWDOWN_PCT = 0.02 # 2% Max Daily Loss
        self.GLOBAL_KILL_SWITCH = False

    def validate_trade(self, account_equity: float, account_balance: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on Risk Rules.
        Returns: (is_allowed: bool, reason: str)
        """
        
        # 0. Global Switch
        if self.GLOBAL_KILL_SWITCH:
            return False, "GLOBAL_KILL_SWITCH_ACTIVE"

        # 1. Daily Drawdown Check
        # specific logic: We need to know the 'Starting Balance' of the day.
        # We store this in StateManager.
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Get start balance from state
        start_balance = self.state.get_state(f"balance_start_{today_str}")
        
        if start_balance is None:
            # First run of the day, set it.
            # CAUTION: Assuming current balance is the start if not found.
            # In production, this might need a more robust sync.
            self.state.set_state(f"balance_start_{today_str}", account_balance)
            start_balance = account_balance
            print(f"[RISK] New Day Detected. Baseline Balance: {start_balance}")

        # Calculate Drawdown
        current_dd_pct = (start_balance - account_equity) / start_balance
        
        if current_dd_pct >= self.MAX_DAILY_DRAWDOWN_PCT:
            self.state.log_event("RISK_VIOLATION", {
                "rule": "MAX_DAILY_DRAWDOWN",
                "current": current_dd_pct,
                "limit": self.MAX_DAILY_DRAWDOWN_PCT
            })
            return False, f"DAILY_DD_LIMIT_HIT ({current_dd_pct:.2%})"

        return True, "OK"

    def set_kill_switch(self, active: bool):
        self.GLOBAL_KILL_SWITCH = active
        self.state.log_event("KILL_SWITCH_CHANGE", {"active": active})
