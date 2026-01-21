from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class InsightDirection(Enum):
    UP = 1
    FLAT = 0
    DOWN = -1

class InsightType(Enum):
    PRICE = 0
    VOLATILITY = 1

@dataclass
class Insight:
    """
    Represents a prediction or signal, based on QuantConnect.Algorithm.Framework.Alphas.Insight
    """
    symbol: str
    generated_time_utc: datetime
    type: InsightType
    direction: InsightDirection
    period: timedelta
    magnitude: Optional[float] = None
    confidence: Optional[float] = None
    weight: Optional[float] = None
    score: float = 0.0

    @staticmethod
    def price(symbol: str, period: timedelta, direction: InsightDirection, magnitude: float = None, confidence: float = None) -> 'Insight':
        return Insight(
            symbol=symbol,
            generated_time_utc=datetime.utcnow(),
            type=InsightType.PRICE,
            direction=direction,
            period=period,
            magnitude=magnitude,
            confidence=confidence
        )

@dataclass
class PortfolioTarget:
    """
    Represents a target holding for a security, based on QuantConnect.Algorithm.Framework.Portfolio.PortfolioTarget
    """
    symbol: str
    quantity: float # Signed quantity (+ for Long, - for Short)
    percent: Optional[float] = None # Target percentage of portfolio equity
