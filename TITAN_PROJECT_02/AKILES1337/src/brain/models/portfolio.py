from typing import List
from ..core.interfaces import PortfolioConstructionModel
from ..core.types import Insight, PortfolioTarget

class EqualWeightingPortfolioConstructionModel(PortfolioConstructionModel):
    """
    Allocates equal capital to all active insights.
    """
    def create_targets(self, insights: List[Insight]) -> List[PortfolioTarget]:
        targets = []
        if not insights:
            return targets
            
        # Simplification: Assume we want to allocate equally among all insights
        # In a real bot, we'd check Active Securities, Cash, etc.
        
        count = len(insights)
        percent = 1.0 / count if count > 0 else 0.0
        
        for insight in insights:
            # Direction affects sign: UP=Positive, DOWN=Negative
            qty_multiplier = 1 if insight.direction.value > 0 else -1
            
            # Create a target (Abstract quantity calculation)
            # We use 'percent' of equity
            targets.append(PortfolioTarget(
                symbol=insight.symbol,
                quantity=0, # Calculated later by Execution
                percent=percent * qty_multiplier
            ))
            
        return targets
