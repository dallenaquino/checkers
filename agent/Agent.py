import random
from abc import ABC, abstractmethod
from typing import Optional

from GameState import GameState


class Agent(ABC):
    @abstractmethod
    def choose_move(self, game_state: GameState):
        pass


class DeterministicAgent(Agent):
    """Always returns the first possible move"""

    def choose_move(self, game_state: GameState):
        possible_moves = game_state.get_legal_moves()
        return possible_moves[0]

class RandomAgent(Agent):
    def choose_move(self, game_state: GameState):
        valid_moves = game_state.get_legal_moves()
        if len(valid_moves) == 0:
            pass
        idx = random.randint(0, len(valid_moves) - 1)
        return valid_moves[idx]


class DecisionTreeNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.moves = []
        self.children = []
        self.game_state: Optional[GameState] = None

# class DecisionTreeAgent(Agent, ABC):
#
#     @classmethod
#     def reorder_moves(cls, moves):
#         return moves