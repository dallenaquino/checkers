import math
from typing import Any, Iterable

from GameState import GameState
from agent.Agent import Agent


class AlphaBetaAgent(Agent):
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def choose_move(self, game_state: GameState, max_depth=math.inf):
        player = game_state.current_player_index
        value, move = self._max_value(game_state, -math.inf, math.inf, player, 0)
        return move

    def _max_value(self, game_state: GameState, alpha, beta, player_index, depth) -> tuple[float, Any]:
        if game_state.is_terminal() or depth >= self.max_depth:
            return self.eval(game_state, player_index), None
        max_val = -math.inf
        best_move = None
        for move in game_state.get_legal_moves():
            val, _ = self._min_value(game_state.generate_successor(move), alpha, beta, player_index, depth + 1)
            if val > max_val:
                max_val = val
                best_move = move
                alpha = max(alpha, val)
                if max_val >= beta:
                    return max_val, best_move
        return max_val, best_move

    def _min_value(self, game_state: GameState, alpha, beta, player_index, depth) -> tuple[float, Any]:
        if game_state.is_terminal() or depth >= self.max_depth:
            return self.eval(game_state, player_index), None
        min_val = math.inf
        best_move = None
        all_moves = game_state.get_legal_moves()
        sorted_moves = self.reorder_moves(all_moves)
        for move in self.reorder_moves(sorted_moves):
            val, _ = self._max_value(game_state.generate_successor(move), alpha, beta, player_index, depth + 1)
            if val < min_val:
                min_val, best_move = val, move
                beta = min(beta, val)
                if min_val <= alpha:
                    return min_val, best_move
        return min_val, best_move

    @classmethod
    def eval(cls, state: GameState, player_index):
        """A basic heuristic evaluation function for a GameState
        Returns 1 if the specified player has won, 0 if someone else has won, and 0.5 otherwise.
        Deriving classes should almost certainly override this method."""
        if state.is_terminal():
            return int(state.get_winner() == player_index)
        return 0.5

    def reorder_moves(self, moves: Iterable):
        return moves
