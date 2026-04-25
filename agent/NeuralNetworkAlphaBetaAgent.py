import random
from typing import Iterable

from CheckersGame import CheckersGameState
from GameState import GameState
from agent.AlphaBetaAgent import AlphaBetaAgent
from agent.CheckersNeuralNetworkAgent import CheckersNeuralNetworkAgent


class NeuralNetworkAgent(AlphaBetaAgent, CheckersNeuralNetworkAgent):
    def choose_move(self, game_state: GameState):
        if game_state.is_terminal():
            return None
        rand = random.random()
        if rand < self.epsilon:
            moves = game_state.get_legal_moves()
            return random.choice(moves)
        return super().choose_move(game_state)

    def __init__(self, file, epsilon=0.2, max_depth=2):
        AlphaBetaAgent.__init__(self, max_depth)
        CheckersNeuralNetworkAgent.__init__(self, file)
        self.epsilon = epsilon

    def eval(self, state: CheckersGameState):
        x = self.prepare_data(state)
        y = self.model.predict(x, verbose=0)
        return y.item()

    def reorder_moves(self, state, moves: Iterable):
        return sorted(moves,
                      key=lambda m: self.eval(state.generate_successor(m)),
                      reverse=not state.current_player_index)