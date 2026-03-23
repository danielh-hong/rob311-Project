"""
Bazaar-AI: A Reinforcement Learning Environment for the Jaipur Card Game

A lightweight simulation framework based on the strategic card game Jaipur,
designed for training and evaluating AI agents.
"""

from .bazaar import Bazaar, BasicBazaar
from .trader import Trader, TraderAction, TraderActionType, SellAction, TakeAction, TradeAction
from .market import Market, MarketObservation
from .goods import GoodType, Goods
from .coins import BonusType, Coins

__version__ = "0.3.0"
__all__ = [
    'Bazaar',
    'BasicBazaar',
    'Trader',
    'TraderAction',
    'TraderActionType',
    'SellAction',
    'TakeAction',
    'TradeAction',
    'Market',
    'MarketObservation',
    'GoodType',
    'Goods',
    'BonusType',
    'Coins',
]
