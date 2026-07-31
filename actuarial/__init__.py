"""Actuarial science modules."""
from .solvency import SolvencyCalculator
from .loss_modeling import LossReserving
from .mortality import MortalityForecaster
__all__ = ["SolvencyCalculator", "LossModeler", "MortalityForecaster"]
__all__ = ["SolvencyCalculator", "LossReserving", "MortalityForecaster"]
