"""Actuarial science modules."""
from .solvency import SolvencyCalculator
from .loss_modeling import LossModeler
from .mortality import MortalityForecaster
__all__ = ["SolvencyCalculator", "LossModeler", "MortalityForecaster"]
