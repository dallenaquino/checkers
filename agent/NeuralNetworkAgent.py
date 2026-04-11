import random
from typing import Iterable

import numpy as np
from tensorflow import keras

from CheckersGame import CheckersGameState, CheckerboardCodes
from GameState import GameState
from agent.Agent import Agent
from util import board_positions


class NeuralNetworkAgent(Agent):
    def choose_move(self, game_state: GameState):
        if game_state.is_terminal():
            return None
        moves = game_state.get_legal_moves()
        rand = random.random()
        if rand < self.epsilon:
            return random.choice(moves)
        x = self.prepare_data(game_state.generate_successor(move) for move in moves)
        y = self.model.predict(x, verbose=0)
        max_idx = np.argmax(y)
        return moves[max_idx]

    def __init__(self, file):
        super().__init__()
        self.epsilon = 0.25
        self.max_depth = 5
        self.model = keras.models.load_model(file)

    @classmethod
    def prepare_data(cls, states: CheckersGameState | Iterable[CheckersGameState]):
        if isinstance(states, CheckersGameState):
            states = [states]
        elif not isinstance(states, list):
            states = list(states)
        x = np.zeros((len(states), 8, 8, 5))
        for idx, state in enumerate(states):
            x[idx,:,:,4] = state.current_player_index
            board = state.board
            for i, j in filter(lambda p: board[p[0]][p[1]] != CheckerboardCodes.DARK_SQUARE, board_positions()):
                code = board[i][j]
                x[idx][i][j][code.value] = 1
        return x
