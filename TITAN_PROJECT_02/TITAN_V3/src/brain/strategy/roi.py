from datetime import timedelta

class ROITable:
    def __init__(self):
        # Format: {minutes_held: profit_target_percent}
        # Example:
        # 0-10 min: 5% (Scalping/Pump)
        # 10-40 min: 3%
        # 40-80 min: 1%
        # >80 min: 0.5% (Just get out with profit)
        self.roi_config = {
            0: 0.05,   # > 5% profit immediately
            10: 0.03,  # > 3% profit after 10 mins
            40: 0.01,  # > 1% profit after 40 mins
            80: 0.005  # > 0.5% profit after 80 mins
        }

    def should_sell(self, entry_time, current_time, current_profit_percent):
        """
        Determines if the trade should be closed based on ROI table.
        """
        duration = (current_time - entry_time).total_seconds() / 60.0 # minutes
        
        # Find the appropriate ROI target for this duration
        target_roi = 0.01 # Default fallback
        
        # Sort keys to iterate correctly
        sorted_times = sorted(self.roi_config.keys(), reverse=True)
        
        for t in sorted_times:
            if duration >= t:
                target_roi = self.roi_config[t]
                break
        
        if current_profit_percent >= target_roi:
            return True, f"ROI Target Reached: {current_profit_percent:.2%} > {target_roi:.2%} in {duration:.1f} min"
            
        return False, ""
