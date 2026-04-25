from abc import ABC
from typing import Iterable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

import numpy as np

from CheckersGame import CheckersGameState, DARK_SQUARE
from agent.Agent import Agent
from util import board_positions


class CheckersNeuralNetworkAgent(Agent, ABC):
    def __init__(self, file):
        self.model = keras.models.load_model(file)


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
            for i, j in filter(lambda p: board[4 * p[0] + p[1] // 2] != DARK_SQUARE, board_positions):
                code = board[4 * i + j // 2]
                x1[idx][i][j][code] = 1
        x2 = np.array([[state.moves_since_capture] for state in states])
        return x1, x2