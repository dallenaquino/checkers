import math
from typing import Any, Iterable

from GameState import GameState
from agent.Agent import Agent


class AlphaBetaAgent(Agent):
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def choose_move(self, game_state: GameState):
        player = game_state.current_player_index
        func = self._min_value if player else self._max_value
        value, move = func(game_state, -math.inf, math.inf, player, 0)
        return move

    def _max_value(self, game_state: GameState, alpha, beta, player_index, depth) -> tuple[float, Any]:
        if game_state.is_terminal() or depth >= self.max_depth:
            return self.eval(game_state), None
        max_val = -math.inf
        best_move = None
        sorted_moves = self.reorder_moves(game_state, game_state.get_legal_moves())
        for move in sorted_moves:
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
            return self.eval(game_state), None
        min_val = math.inf
        best_move = None
        all_moves = game_state.get_legal_moves()
        sorted_moves = self.reorder_moves(game_state, all_moves)
        for move in sorted_moves:
            val, _ = self._max_value(game_state.generate_successor(move), alpha, beta, player_index, depth + 1)
            if val < min_val:
                min_val, best_move = val, move
                beta = min(beta, val)
                if min_val <= alpha:
                    return min_val, best_move
        return min_val, best_move


    def eval(self, state: GameState):
        """A basic heuristic evaluation function for a GameState
        Returns 1 if player1 has won, -1 if player2 has won, and 0 otherwise.
        Deriving classes should almost certainly override this method."""
        if state.is_terminal() and (winner := state.get_winner()) is not None:
            return -1 ** winner
        return 0

    def reorder_moves(self, state, moves: Iterable):
        return moves
