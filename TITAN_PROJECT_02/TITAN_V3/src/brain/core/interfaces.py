from abc import ABC, abstractmethod
from typing import List
from .types import Insight, PortfolioTarget

class AlphaModel(ABC):
    """
    Abstract Base Class for Alpha Models.
    Responsibility: Generate Insights (Predictions) from Data.
    """
    def __init__(self, name: str = "GenericAlpha"):
        self.name = name

    @abstractmethod
    def update(self, data) -> List[Insight]:
        """
        Updates the model with new data and returns generated insights.
        """
        pass

class PortfolioConstructionModel(ABC):
    """
    Abstract Base Class for Portfolio Construction.
    Responsibility: Convert Insights into PortfolioTargets (Allocation).
    """
    @abstractmethod
    def create_targets(self, insights: List[Insight]) -> List[PortfolioTarget]:
        """
        Determines target portfolio allocations based on insights.
        """
        pass

class RiskManagementModel(ABC):
    """
    Abstract Base Class for Risk Management.
    Responsibility: Adjust PortfolioTargets to ensure safety (Stop Loss, Max Drawdown).
    """
    @abstractmethod
    def manage_risk(self, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        """
        Adjusts targets to meet risk constraints.
        """
        pass
