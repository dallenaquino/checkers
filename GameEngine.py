from abc import ABC, abstractmethod
from typing import Optional

from GameState import GameState
from agent import Agent


class GameEngine(ABC):
    def __init__(self, *agents):
        self.current_state: Optional[GameState] = None
        self._num_players = 2
        self._agents: list[Agent] = [*agents]

    def play_game(self):
        while not self.current_state.is_terminal():
            move = self.get_next_move()
            new_state = self.make_move(move)
            self.current_state = new_state

    def get_next_move(self):
        current_player_idx = self.current_state.current_player_index
        move = self._agents[current_player_idx].choose_move(self.current_state)
        return move

    @abstractmethod
    def make_move(self, move):
        pass
