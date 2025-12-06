import itertools
from typing import Iterable

from CheckersGame import CheckersGameState, CheckerboardCodes, PromotionJumpMove, JumpMove, PushMove, PromotionPushMove, \
    CheckersMove
from agent.AlphaBetaAgent import AlphaBetaAgent


class CheckersAlphaBetaAgent(AlphaBetaAgent):
    @classmethod
    def eval(cls, state: CheckersGameState, player_index=0):
        if state.is_terminal():
            return 1 if state.get_winner() == player_index else -1
        board = state.board
        dark_squares = tuple((i, j) for i, j in itertools.permutations(range(8), 2) if (i + j) % 2)
        red_regular = tuple(filter(lambda pos: board[pos[0]][pos[1]] == CheckerboardCodes.RED_PIECE, dark_squares))
        red_king = tuple(filter(lambda pos: board[pos[0]][pos[1]] == CheckerboardCodes.RED_KING, dark_squares))
        black_regular = tuple(filter(lambda pos: board[pos[0]][pos[1]] == CheckerboardCodes.BLACK_PIECE, dark_squares))
        black_king = tuple(filter(lambda pos: board[pos[0]][pos[1]] == CheckerboardCodes.BLACK_KING, dark_squares))
        value = 2 * (len(black_king) - len(red_king)) + (len(black_regular) - len(red_regular))
        if state.current_player_index != player_index:
            value *= -1
        total_pieces = len(red_regular) + len(red_king) + len(black_regular) + len(black_king)
        return value / total_pieces

    def reorder_moves(self, moves: Iterable[CheckersMove]) -> Iterable[CheckersMove]:
        def move_priority(move: CheckersMove):
            final_coord = move.compute_final_coord()
            priority = 0
            if isinstance(move, JumpMove):
                priority -= 2
                priority -= len(move.actions)
                if isinstance(move, PromotionJumpMove):
                    priority -= 2
            elif isinstance(move, PushMove):
                if isinstance(move, PromotionPushMove):
                    priority -= 1
            else:
                return 5
            if final_coord[1] in (0, 7):
                priority -= 1
            return priority

        return list(sorted(moves, key=move_priority))