import random
from typing import Iterable

import numpy as np

from CheckersGame import CheckersGameState, CheckerboardCodes
from GameState import GameState
from agent.CheckersNeuralNetworkAgent import CheckersNeuralNetworkAgent
from util import board_positions


class GreedyNeuralNetworkAgent(CheckersNeuralNetworkAgent):
    def choose_move(self, game_state: GameState):
        if game_state.is_terminal():
            return None
        rand = random.random()
        moves = game_state.get_legal_moves()
        if rand < self.epsilon:
            return random.choice(moves)
        x = self.prepare_data(game_state.generate_successor(move) for move in moves)
        x = x[:len(self.model.inputs)]
        if len(x) == 1:
            x = x[0]
        y = self.model.predict(x, verbose=0)
        max_idx = np.argmax(y) if game_state.current_player_index == 0 else np.argmin(y)
        return moves[max_idx]

    def __init__(self, file, epsilon=0.2):
        super().__init__(file)
        self.epsilon = epsilon

    @classmethod
    def prepare_data(cls, states: CheckersGameState | Iterable[CheckersGameState]):
        if isinstance(states, CheckersGameState):
            states = [states]
        elif not isinstance(states, list):
            states = list(states)
        x1 = np.zeros((len(states), 8, 8, 5))
        for idx, state in enumerate(states):
            x1[idx,:,:,4] = state.current_player_index
            board = state.board
            for i, j in filter(lambda p: board[4 * p[0] + p[1] // 2] != CheckerboardCodes.DARK_SQUARE, board_positions):
                code = board[4 * i + j // 2]
                x1[idx][i][j][code] = 1
        x2 = np.array([[state.moves_since_capture] for state in states])
        return x1, x2