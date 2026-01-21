from typing import List, Optional
from datetime import datetime, timedelta
from ..core.interfaces import AlphaModel
from ..core.types import Insight, InsightDirection, InsightType
from ..strategy.roi import ROITable

class ROIAlphaModel(AlphaModel):
    def __init__(self, name="ROI_Logic_v1"):
        super().__init__(name=name)
        self.roi_engine = ROITable()

    def update(self, data) -> List[Insight]:
        """
        Checks if the current position should be closed based on ROI targets.
        Returns an Insight(FLAT) if criteria met.
        """
        if not data.has_position:
            return []

        # Convert timestamp to datetime
        entry_dt = datetime.fromtimestamp(data.open_time)
        
        # Calculate Profit % (Defensive calculation)
        profit_pct = 0.0
        current_price = data.bid if data.position_type == 0 else data.ask
        
        if data.open_price > 0.0001:
            if data.position_type == 0: # Buy
                profit_pct = (current_price - data.open_price) / data.open_price
            else: # Sell
                profit_pct = (data.open_price - current_price) / data.open_price
        
        should_close, reason = self.roi_engine.should_sell(entry_dt, datetime.now(), profit_pct)
        
        if should_close:
            # Generate Flat Insight (Close Signal)
            # Duration 1 min, High Confidence
            # Magnitude 0.0 implies no directional conviction (Close/Flat)
            return [Insight.price(
                symbol=data.symbol, 
                period=timedelta(minutes=1), 
                direction=InsightDirection.FLAT, 
                magnitude=0.0, 
                confidence=1.0
            )]
            
        return []
