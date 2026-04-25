from abc import ABC, abstractmethod
from typing import Optional, Any

from GameState import GameState, MoveInfo
from agent import Agent


class GameEngine(ABC):
    def __init__(self, *agents):
        self.current_state: Optional[GameState] = None
        self._num_players = 2
        self._agents: list[Agent] = [*agents]

    def play_game(self, state_logger: Any = None):
        while not self.current_state.is_terminal():
            if state_logger is not None:
                state_logger.log(self.current_state)
            move = self.get_next_move()
            self.make_move(move)

        if state_logger is not None:
            winner = self.current_state.get_winner()
            score = 0 if winner is None else (-1) ** winner
            state_logger.log(self.current_state, score)

    def get_next_move(self) -> MoveInfo:
        current_player_idx = self.current_state.current_player_index
        move = self._agents[current_player_idx].choose_move(self.current_state)
        return move

    @abstractmethod
    def make_move(self, move: MoveInfo):
        pass
